#!/bin/bash

# Render startup script for Eye Disease Detection API

echo "Starting Eye Disease Detection API on Render..."

# Start the application with Gunicorn
exec gunicorn --config gunicorn.conf.py app:app
