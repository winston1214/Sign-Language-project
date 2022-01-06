import numpy as np
from tqdm.notebook import tqdm
import re

def video_sampling(data,video_size):
    X_train=[]
    num_ls = [0]

    hand_frame_ls = np.array([])


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

            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())

        if num_ls[idx+1] - num_ls[idx] <0 or idx == len(data)-1: # 동영상이 바뀔 때
            hand_frame_ls = hand_frame_ls.reshape(-1,246)
            if len(hand_frame_ls)<video_size: # 손 탐지된 횟수가 정한 video 개수보다 적을 때
                end_frame = np.array([hand_frame_ls[-1]]) # 끝 프레임
                tmp = np.repeat(end_frame,video_size - len(hand_frame_ls),axis=0) # 끝 프레임 반복

                hand_frame_ls = np.append(hand_frame_ls,tmp).reshape(-1,246)
            elif len(hand_frame_ls) > video_size:
                hand_frame_ls = hand_frame_ls[:video_size]

            X_train.append(hand_frame_ls.tolist())
            hand_frame_ls = np.array([])
    X_train = np.array(X_train).reshape(-1,video_size,246)
    return X_train
