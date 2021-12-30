import json
from tqdm import tqdm
import re
import numpy as np
with open('AlphaPose_json/alphapose-results.json','r') as f: # 수정
    data = json.load(f)

video_size = 80 # 사용할 frame 수
X_train=[]
num_ls = [0]
new_X_train = []
result = []
hand_frame_ls = np.array([])
video_len = []
for idx,i in tqdm(enumerate(data)):
    num2 = i.split('_')[3]
    num2 = re.sub(".jpg", "", num2)
    num_ls.append(int(num2))

    dt = data[i]['keypoints']
    dt = np.array(dt).reshape(123,3)

    mean_x,std_x = np.mean(dt[:,0]),np.std(dt[:,0]) # noramlization
    mean_y,std_y = np.mean(dt[:,1]),np.std(dt[:,1]) # normalization
    normalization_x, normalization_y = (dt[:,0] - mean_x)/std_x, (dt[:,1] - mean_y)/std_y # normalization
    dt[:,0] = normalization_x # normalization
    dt[:,1] = normalization_y # normalization

    if set([i for i in range(81,124)]) & set(np.where(dt[:,2]>0.5)[0]): # 손이 탐지된 frame

        hand_frame_ls = np.append(hand_frame_ls,dt[:,:1].flatten())

    if num_ls[idx+1] - num_ls[idx] <0 or idx == len(data)-1:
        if len(hand_frame_ls)<video_size: 
            tmp = np.repeat(hand_frame_ls[-123:],video_size-(len(hand_frame_ls) // 123))
            print(len(tmp))
            hand_frame_ls = np.append(hand_frame_ls,tmp)
        X_train.append(hand_frame_ls.tolist())