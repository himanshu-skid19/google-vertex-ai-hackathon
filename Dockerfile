# Use the official Python image from the Docker Hub, specifying the architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory
WORKDIR /app

# Install build dependencies including ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the image
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (assuming Chainlit uses 8000)
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "app.py", "-w"]