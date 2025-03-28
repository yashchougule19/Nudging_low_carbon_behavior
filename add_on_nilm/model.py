import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
import matplotlib.pyplot as plt
from flask import Flask, send_file, CORS
import io
from flask_cors import CORS

# ============================
# Set Up Flask Server
# ============================
app = Flask(__name__)
CORS(app)  # Enable CORS to allow external access

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================
# Load Trained NILM Model and Novel Data
# ============================

@app.route('/favicon.ico')
def favicon():
    return ('', 204)  # Respond with No Content

def get_novel_data_from_ha(db_path):
    """Extracts Home Assistant energy data and preprocesses it for NILM prediction."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT 
                s.last_updated_ts,
                DATETIME(s.last_updated_ts, 'unixepoch', 'localtime') AS last_updated_localtime,
                s.state
            FROM states AS s
            LEFT JOIN states_meta AS sm ON s.metadata_id = sm.metadata_id
            WHERE sm.entity_id = "sensor.tasmota_energy_current_consumption_w"
            """
            ha_data = pd.read_sql_query(query, conn)

        if ha_data.empty:
            logging.error("No data retrieved from the database.")
            return None

        ha_data['last_updated_localtime'] = pd.to_datetime(ha_data['last_updated_localtime'])
        ha_data['state'] = pd.to_numeric(ha_data['state'], errors='coerce')
        ha_data = ha_data.sort_values(by='last_updated_localtime')

        ha_data.set_index('last_updated_localtime', inplace=True)
        ha_data_resampled = ha_data.resample('15T').mean()
        ha_data_resampled['state_cleaned'] = ha_data_resampled['state'].interpolate(method='linear')

        ha_data_resampled['time_diff_hours'] = 0.25  # 15 minutes = 0.25 hours
        ha_data_resampled['energy_kwh'] = (ha_data_resampled['state_cleaned'] / 1000) * ha_data_resampled['time_diff_hours']
        ha_data_resampled['hour'] = ha_data_resampled.index.hour
        ha_data_resampled['day_of_week'] = ha_data_resampled.index.dayofweek

        ha_data_resampled.rename(columns={'state_cleaned': 'Aggregate'}, inplace=True)
        ha_data_resampled.reset_index(inplace=True)

        novel_data = ha_data_resampled[['last_updated_localtime', 'hour', 'day_of_week', 'Aggregate', 'energy_kwh']].dropna()
        logging.info(f"Extracted {len(novel_data)} records from Home Assistant database.")

        return novel_data
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")
    return None

def predict_nilm(novel_data, model_path):
    """Loads trained model and predicts NILM values."""
    try:
        rf_model = joblib.load(model_path)
        predictions = rf_model.predict(novel_data)
        return predictions
    except Exception as e:
        logging.error(f"Error loading or predicting with the model: {e}")
        return None

# Define database and model paths
db_path = "/config/home-assistant_v2.db"
model_path = "/config/nilm_rf_model.pkl"  

# Extract and preprocess novel data
novel_data = get_novel_data_from_ha(db_path)
results_last_day = None

if novel_data is not None and not novel_data.empty:
    features = ['hour', 'day_of_week', 'Aggregate', 'energy_kwh']
    targets = ['Appliance1', 'Appliance2', 'Appliance3', 'Appliance4', 'Appliance5']

    novel_data_prepared = novel_data[features]
    predictions = predict_nilm(novel_data_prepared, model_path)

    if predictions is not None:
        results = pd.DataFrame(novel_data[['last_updated_localtime', 'Aggregate']])
        results[targets] = predictions
        results['last_updated_localtime'] = pd.to_datetime(results['last_updated_localtime'])
        results.set_index('last_updated_localtime', inplace=True)

        last_day = results.index[-1].date()
        results_last_day = results[results.index.date == last_day]
    else:
        logging.error("Predictions could not be generated.")
else:
    logging.error("No valid data available for NILM processing.")

# ============================
# Serve the Homepage using Flask
# ============================

@app.route('/')
def home():
    return "NILM Flask API is running. Use /plot to see the graph."

# ============================
# Serve the Plot using Flask
# ============================

@app.route('/plot')
def serve_plot():
    """Generates and serves the energy consumption plot."""
    if results_last_day is None or results_last_day.empty:
        return "Error: No data available for plotting.", 500

    fig, ax = plt.subplots(figsize=(14, 7))

    results_last_day['Aggregate'].plot(ax=ax, label='Main Meter', color='black', linewidth=2)
    for appliance in targets:
        results_last_day[appliance].plot(ax=ax, label=f'Predicted - {appliance}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy (W)')
    ax.set_title(f'Energy Consumption: Main Meter + Predicted Appliances (15-Minute Intervals)')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

# ============================
# Run Flask Server
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
