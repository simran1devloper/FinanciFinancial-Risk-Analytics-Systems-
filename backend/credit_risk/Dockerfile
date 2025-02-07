# Use an official Python image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py credit-risk_ana2.py best_xgb_model.json ./

# Expose the port the app will run on
EXPOSE 8006

# Run the command to start the app when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8006"]