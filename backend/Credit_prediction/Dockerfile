# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirement.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the application code
COPY . .

# Command to run the application
CMD ["uvicorn", "credit_main:app", "--host", "0.0.0.0", "--port", "8007"]