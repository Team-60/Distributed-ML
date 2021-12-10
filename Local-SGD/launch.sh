#!/bin/bash

python3 main.py --world-size 3 --rank 0 & pid=$!
PID_LIST+=" $pid";
python3 main.py --world-size 3 --rank 1 & pid=$!
PID_LIST+=" $pid";
python3 main.py --world-size 3 --rank 2 & pid=$!
PID_LIST+=" $pid";

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";


