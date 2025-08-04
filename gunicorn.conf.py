# Gunicorn configuration file for Render deployment

import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = 1  # Single worker to avoid memory issues with large ML models
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for model loading and predictions
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "eye-disease-detection"

# Server mechanics
preload_app = False  # Don't preload to allow lazy loading
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in future)
keyfile = None
certfile = None
