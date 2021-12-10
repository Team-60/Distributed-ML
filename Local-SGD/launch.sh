#!/bin/bash

python3 downloader.py

for ((i = 0; i < $1; i ++))
do 
    python3 main.py --world-size $1 --rank $i & pid=$!
    PID_LIST+=" $pid";
done

trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";


