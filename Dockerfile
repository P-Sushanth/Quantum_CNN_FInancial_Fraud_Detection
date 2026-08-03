# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port and start
# Hugging Face Spaces usually listens on 7860
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
