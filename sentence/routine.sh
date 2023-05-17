#!/bin/bash
source /data/wujk/wujk_env/bin/activate
if [ $? -ne 0 ]
then
    echo "failed"
    source /home/data/wujk/wujk_env/bin/activate
    cd /home/data/wujk/codes/SimCSE/bash
else
    echo "YES"
    cd /data/wujk/codes/SimCSE/bash
fi