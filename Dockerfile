# Use a lightweight Python base image
FROM python:3.9-slim

# Set an environment variable to distinguish Docker environment
ENV DOCKER_ENV=true

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (optimized for Docker caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . . 

# Ensure documents folder is copied explicitly (if not already covered by COPY .)
# Not strictly required if COPY . copies the entire project, but for clarity:
COPY documents /app/documents

# Use Gunicorn for a production-ready WSGI server
RUN pip install gunicorn

# Expose the port the app will run on
EXPOSE 5000

# Use Gunicorn to run the Flask app in production mode
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
