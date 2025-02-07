# Use an official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirement.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the application code
COPY new_main.py new_forest.pkl new_scaler.pkl ./

# Expose the port the app will run on
EXPOSE 8005

# Run the command to start the app when the container launches
CMD ["uvicorn", "new_main:app", "--host", "0.0.0.0", "--port", "8005"]