#!/usr/bin/env bash
set -euo pipefail

# Run script with automatic cleanup of stuck OpenOCD sessions
# This will kill any existing OpenOCD processes and their ports

echo "Checking for existing OpenOCD sessions..."

# Kill any existing OpenOCD processes
if pgrep -x openocd > /dev/null; then
    echo "Found existing OpenOCD process(es), killing them..."
    pkill -9 openocd || true
    sleep 1
    echo "Killed existing OpenOCD sessions"
fi

# Also check and free up the common ports
for port in 3333 4444 6666; do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "Freeing up port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    fi
done

echo ""
echo "Starting OpenOCD with semihosting..."
echo "Press Ctrl+C to stop"
echo "======================================"

# Start OpenOCD with semihosting enabled
# This will show debug printf output in the terminal
openocd \
	-f interface/stlink.cfg \
	-f target/stm32h7x.cfg \
	-c "init" \
	-c "arm semihosting enable" \
	-c "reset run"
