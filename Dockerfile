# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . .

# Expose port
EXPOSE 8080

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
