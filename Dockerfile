# Start with a lean official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install all system dependencies required by our libraries
# This prevents the environment errors we saw before
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the Playwright browser binaries
RUN playwright install

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# The command to run your application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]