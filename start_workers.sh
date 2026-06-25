#!/bin/bash
source ~/myenv/bin/activate
cd ~/multi_agent_benchmark/agents

echo "Starting 3 workers..."

python worker.py worker_1 8001 &
python worker.py worker_2 8002 &
python worker.py worker_3 8003 &

echo "All 3 workers started!"
echo "Workers running on ports 8001, 8002, 8003"
echo "To stop all workers run: pkill -f worker.py"