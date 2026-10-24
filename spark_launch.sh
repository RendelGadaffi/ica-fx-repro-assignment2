#!/bin/bash
# Launch the ICA-FX full suite on Sparky, detached, then exit fast so the ssh
# session can die without taking the run with it. Self-safe: the pkill pattern
# ("[r]un_all") cannot match this script's own command line.
cd /home/rendel/ica-fx || exit 1
pkill -f "[r]un_all.py" 2>/dev/null
sleep 1
rm -f run2.log
OMP_NUM_THREADS=1 setsid nohup ./venv/bin/python /home/rendel/ica-fx/run_all.py --datasets linear nonlinear boston oj --jobs 20 > /home/rendel/ica-fx/run2.log 2>&1 < /dev/null &
sleep 6
ls -la /home/rendel/ica-fx/run2.log
wc -l /home/rendel/ica-fx/run2.log
echo SPARKY_LAUNCH_OK
