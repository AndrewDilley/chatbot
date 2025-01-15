# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Reference an environment variable passed during runtime

# Copy necessary files into the container
COPY requirements.txt /app/
COPY app.py /app/
COPY config.py /app/

# Copy the templates directory for HTML files
COPY templates /app/templates/

# Copy the static directory for CSS and JS files
COPY static /app/static/


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FAISS index and text map files into the container
COPY preprocessed_data/faiss_index.bin /app/preprocessed_data/
COPY preprocessed_data/text_map.json /app/preprocessed_data/

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables
ENV DOCKER_ENV=true

# Run the app
CMD ["python", "app.py"]
