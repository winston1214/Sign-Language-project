import os
import shutil
ls = os.listidir('/workspace/sign_data')
for i in ls:
    if len(i) != 22:
        os.rename('/workspace/sign_data/'+i,'/workspace/sign_data/'+i[-22:])
video_num = list(map(lambda x: int(x.split('_')[-1].split('.')[0])))
part1 = max(video_num)//3
part2 = max(video_num)//3 * 2
os.mkdir('/workspace/sign_data/{}_{}'.format(min(video_num),part1))
os.mkdir('/workspace/sign_data/{}_{}'.format(part1+1,part2))
os.mkdir('/workspace/sign_data/{}_{}'.format(part2+1,max(video_num)))
for i in ls:
    video_number = int(i.split('_')[-1].split('.')[0])
    if video_number <= part1:
        shutil.move('/workspace/sign_data/'+i,'/workspace/sign_data/{}_{}/{}'.format(min(video_num),part1,i))
    if part1<video_number<=part2:
        shutil.move('/workspace/sign_data/'+i,'/workspace/sign_data/{}_{}/{}'.format(part1+1,part2,i))
    if part2<video_number:
        shutil.move('/workspace/sign_data/'+i,'/workspace/sign_data/{}_{}/{}'.format(part2+1,max(video_num),i))