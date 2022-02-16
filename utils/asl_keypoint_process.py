import os
import numpy as np
import json
from tqdm.notebook import tqdm
import math
import os
def asl_keti(path):
    np.random.seed(42)
    if path[-1] != '/':
        path += '/'
    json_ls = os.listdir(path)
    X_data = np.array([])
    max_frame_num = 16
    n = max_frame_num
    for name in tqdm(sorted(json_ls)):
        with open(path+name,'r') as f:
            data = json.load(f)
        video_key = np.array([])
        num_ls = []

        for i in data:
            num = i.split('.')[0].split('_')[-1]
            num_ls.append(int(re.sub("images", "",num)))
            dt = data[i]['keypoints']
            dt = np.array(dt).reshape(123,3)
            dt = np.delete(dt,range(13,81),axis=0) # 얼굴 키포인트 제외
            dt = dt[:,:2]
            x,y = dt[:,0],dt[:,1]
            mean_x, mean_y = np.mean(x),np.mean(y)
            std_x,std_y = np.std(x),np.std(y)
            normal_x,normal_y = (x-mean_x)/std_x , (y - mean_y)/std_y
            dt[:,0] = normal_x
            dt[:,1] = normal_y
            video_key = np.append(video_key,dt)
        max_num = max(num_ls)
        start_num = min(num_ls)
        video_key = video_key.reshape(-1,110)
        check_ls = [i for i in range(start_num,max_num+1)]
        div = list(set(check_ls) - set(num_ls))

        if div: # alphapose check
            div = sorted(div)
            for ap in div:
                avg = np.mean([video_key[ap-1],video_key[ap]],axis=0)
                video_key = np.insert(video_key,ap,avg,axis=0)
        l = len(data)
        z = math.floor(l/(n-1))
        y = math.floor((l-z*(n-1))/2) # start point
        r = np.random.randint(1,z+1,n)
        baseline = []

        for i in range(n):
            baseline.append(y+i*z)
        select_frame = np.array(baseline) + r
        select_frame[np.where(select_frame >= len(video_key))] = len(video_key)-1
        video_key = video_key[select_frame]
        X_data = np.append(X_data,video_key)
    X_data = X_data.reshape(-1,n,110)