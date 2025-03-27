# Use an official Python image as base
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy files
COPY app.py model.pkl ./

# Install Flask and required libraries
RUN pip install flask pandas numpy scikit-learn

# Expose port 5000
EXPOSE 5000

# Run the Flask API
CMD ["python", "app.py"]

# Install system dependencies (including curl)
RUN apt-get update && apt-get install -y curl
