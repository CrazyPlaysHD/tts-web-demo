echo 'Stopping Web App ...'
kill -9 `cat save_pid_intel.txt`
rm save_pid_intel.txt
echo 'Web App stoped!!!'
