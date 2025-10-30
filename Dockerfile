# Use the Python 3.11-slim base image (matches your libraries)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set the port Hugging Face Spaces expects
ENV PORT=7860
EXPOSE 7860

# Create the directories your app.py needs *before* installing
# 'Uploads' for image uploads
# 'flask_session' for server-side sessions (SESSION_TYPE="filesystem")
RUN mkdir -p /app/Uploads /app/flask_session

# Copy the requirements file first for efficient caching
COPY requirements.txt .

# Install dependencies
# We add 'gunicorn' here because it's the correct production
# server for a Flask app (app.py).
RUN pip install --no-cache-dir --upgrade -r requirements.txt gunicorn

# Copy your entire project's code into the container
# This includes app.py, main.py, src/, api/, templates/, etc.
COPY . .

# The command to run your Flask app using Gunicorn
# It looks for the 'app' variable inside the 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]