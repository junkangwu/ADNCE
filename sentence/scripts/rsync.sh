
timeout_s="1000s"
echo "LDS9"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
timeout $timeout_s rsync -av --exclude-from='/data/wujk/codes/SimCSE/scripts/exclude_files.txt' -e 'ssh -p 2123' /data/wujk/codes/SimCSE wujk@210.45.123.248:/data/wujk/codes/ # lds6
if [[ $? == 124 ]];then
     echo 'LDS9 访问超时'          
    #  exit        # 程序退出
 fi

echo "LDS10"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
timeout $timeout_s rsync -av --exclude-from='/data/wujk/codes/SimCSE/scripts/exclude_files.txt' -e 'ssh -p 2124' /data/wujk/codes/SimCSE wujk@210.45.123.248:/data/wujk/codes/ # lds6
if [[ $? == 124 ]];then
     echo 'LDS9 访问超时'          
    #  exit        # 程序退出
fi

echo "LDS11"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
timeout $timeout_s rsync -av --exclude-from='/data/wujk/codes/SimCSE/scripts/exclude_files.txt' -e 'ssh -p 2125' /data/wujk/codes/SimCSE wujk@210.45.123.248:/data/wujk/codes/ # lds6
if [[ $? == 124 ]];then
     echo 'LDS9 访问超时'          
    #  exit        # 程序退出
fi

echo "LDS6"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
timeout $timeout_s rsync -av --exclude-from='/data/wujk/codes/SimCSE/scripts/exclude_files.txt' -e 'ssh -p 22' /data/wujk/codes/SimCSE wujk@172.16.251.10:/data/wujk/codes/ # lds6
if [[ $? == 124 ]];then
     echo 'LDS6 访问超时'          
    #  exit        # 程序退出
fi