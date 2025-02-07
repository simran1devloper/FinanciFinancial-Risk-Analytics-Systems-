# Use an official Python image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY var.py .

# Expose the port the app will run on
EXPOSE 8008

# Run the command to start the app when the container launches
CMD ["uvicorn", "var:app", "--host", "0.0.0.0", "--port", "8008"]