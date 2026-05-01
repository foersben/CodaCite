#!/bin/bash
# Skill Description: Finds and kills any lingering process on a given port (default: 8080).
# Usage: ./free_port.sh [PORT]

PORT=${1:-8080}
echo "Checking for processes on port $PORT..."

PID=$(fuser $PORT/tcp 2>/dev/null)

if [ -z "$PID" ]; then
    echo "Port $PORT is free. Ready for Uvicorn."
else
    echo "Found blocking process(es): $PID. Terminating..."
    fuser -k $PORT/tcp
    echo "Port $PORT successfully freed."
fi
