import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Printout")
    parser.add_argument("--date", type=str, default='2022-08-31', help="[concat, mean, sum, final]")
    return parser.parse_args()
args = parse_args()
# print(args.date)
f_out = open("/data/wujk/codes/SimCSE/log_all/outputs_{}.txt".format(args.date), 'w')
file_path = "/data/wujk/codes/SimCSE/log_all/{}/".format(args.date)
dirlist = os.listdir(file_path)
dirlist.sort()

filelist = []
file_name = []
score_all = []
score_all_ndcg = []
for file_sin in dirlist:
    file_add = file_path + file_sin
    if os.path.isfile(file_add):
        finish_flag = 0
        cnt = 0
        with open(file_add) as f:
            for line in f.readlines():
                line = line.strip()
                cnt += 1
                if cnt == 5:
                    finish_flag = 1
                    performance = line
                    performance = performance.replace("|", "\t")
        model_name = file_sin[:-4]
        finish_str = "finish" if finish_flag == 1 else "unfinish"
        print_text = "{} {}\n".format(model_name, performance)
        f_out.write(print_text)
        # print(print_text)
f_out.close()