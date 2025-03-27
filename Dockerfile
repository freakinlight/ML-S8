# Use an official Python image as base
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy local files into the container
COPY ecg.csv train_model.py ./

# Install necessary dependencies
RUN pip install pandas scikit-learn imbalanced-learn

# Run the training script
CMD ["python", "train_model.py"]

