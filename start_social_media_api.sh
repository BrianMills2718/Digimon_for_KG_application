#!/bin/bash
# Start the social media analysis API

echo "Starting DIGIMON Social Media Analysis API..."

# Kill any existing instances
pkill -f social_media_api_traced.py 2>/dev/null || true

# Wait a moment for processes to die
sleep 1

# Start the API server
cd /home/brian/digimon_cc
python social_media_api_traced.py