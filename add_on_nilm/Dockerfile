FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY model.py .
COPY run.sh .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the script is executable
RUN chmod +x run.sh

# Set the entrypoint to the script
CMD ["/bin/bash", "/app/run.sh"]
