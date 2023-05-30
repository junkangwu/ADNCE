source /data/wujk/wujk_env/bin/activate
if [ $? -ne 0 ]
then
    echo 'LDS14'
    source /home/sist/jkwu0909/wujk_env/bin/activate
    cd /home/sist/jkwu0909/codes/ADNCE/bash/
else
    echo 'LDS1-LDS13'
    cd /data/wujk/codes/ADNCE/bash/
fi
