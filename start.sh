echo 'Starting Flask Web App ...'
nohup python3.6 app.py &
echo $! > save_pid_intel.txt
echo 'Web App intent started!!!'