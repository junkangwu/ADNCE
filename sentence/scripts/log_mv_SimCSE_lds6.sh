# LDS2
echo "LDS6"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
echo $time_cnt
files=`ls /data/wujk/codes/SimCSE/logs`
new_out_dir="/data/wujk/codes/SimCSE/logs/${time_cnt}/"
if [ ! -d $new_out_dir ]
then
    mkdir $new_out_dir
fi
# echo $files
for file in $files
do
    # echo $file
    if test -f "/data/wujk/codes/SimCSE/logs/${file}"
    then
        # echo "is file"
        cnt_file="/data/wujk/codes/SimCSE/logs/${file}"
        if [ `grep -c "test" ${cnt_file}` -ne '0' ];then
            echo "${cnt_file} Has test!"
            rsync -av -e 'ssh -p 22' $cnt_file "wujk@172.16.251.41:/data/wujk/codes/SimCSE/log_all/${time_cnt}/"
            if [ $? -ne 0 ]
            then
                echo "failed"
            else
                mv $cnt_file $new_out_dir
            fi
        else
            echo "${cnt_file} Has NOT TEST!!"
        fi
        
    # else
        # echo "is directory"
    fi
done