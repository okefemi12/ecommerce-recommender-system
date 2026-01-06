#!/bin/bash
# Start Backend
# FIX 1: Point to 'src.main:app' because main.py is inside the src folder
# FIX 2: Removed '&' so the container stays alive
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
