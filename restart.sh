echo 'Stopping Web App ...'
kill -9 `cat save_pid_intel.txt`
rm save_pid_intel.txt
echo 'Web App stoped!!!'
echo 'Starting Flask Web App ...'
nohup python3.6 app.py &
echo $! > save_pid_intel.txt
echo 'Web App intent started!!!'