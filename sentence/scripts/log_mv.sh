echo "LDS9"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
echo $time_cnt
login3="wujk@210.45.123.248 -p 2123"
timeout 20s ssh $login3 "bash /data/wujk/codes/SimCSE/scripts/log_mv_SimCSE.sh"

echo "LDS10"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
echo $time_cnt
login3="wujk@210.45.123.248 -p 2124"
timeout 20s ssh $login3 "bash /data/wujk/codes/SimCSE/scripts/log_mv_SimCSE.sh"

echo "LDS11"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
echo $time_cnt
login3="wujk@210.45.123.248 -p 2125"
timeout 20s ssh $login3 "bash /data/wujk/codes/SimCSE/scripts/log_mv_SimCSE.sh"

echo "LDS6"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
echo $time_cnt
login3="wujk@172.16.251.10 -p 22"
timeout 20s ssh $login3 "bash /data/wujk/codes/SimCSE/scripts/log_mv_SimCSE_lds6.sh"

# LDS2
echo "LDS7"
echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S)
time_cnt=`echo $(TZ=UTC-8 date +%Y-%m-%d)`
time_cnt2=`echo $(TZ=UTC-8 date +%Y-%m-%d-%H:%M:%S)`
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
            rsync -a $cnt_file "/data/wujk/codes/SimCSE/log_all/${time_cnt}/"
            mv $cnt_file $new_out_dir
        else
            echo "${cnt_file} Has NOT TEST!!"
        fi
        
    # else
        # echo "is directory"
    fi
done

# logout
python print_pos_out_result.py --date $time_cnt