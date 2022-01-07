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


def video_sampling2(data): # original video에서 hand video 부분만 random하게 frame 증강
    np.random.seed(42)
    X_train=np.array([])
    num_ls = [0]
    max_frame_num = 0
    hand_frame_ls = np.array([])
    for idx,i in tqdm(enumerate(data)):
        num2 = i.split('_')[3]
        num2 = re.sub(".jpg", "", num2)
        num_ls.append(int(num2))

        if num_ls[idx+1] - num_ls[idx] <0 or idx == len(data)-1: # 동영상이 바뀔 때
            if max_frame_num < num_ls[idx]:
                max_frame_num = num_ls[idx]
    
    num_ls = [0]
    video_hand_idx = []
    print(max_frame_num)
    video_frame = np.array([])
    
    for idx,i in tqdm(enumerate(data)):
        num2 = i.split('_')[3]
        num2 = re.sub('.jpg','',num2)
        num_ls.append(int(num2))
        dt = data[i]['keypoints']
        dt = np.array(dt).reshape(123,3)
        mean_x,std_x = np.mean(dt[:,0]),np.std(dt[:,0]) # noramlization
        mean_y,std_y = np.mean(dt[:,1]),np.std(dt[:,1]) # normalization
        normalization_x, normalization_y = (dt[:,0] - mean_x)/std_x, (dt[:,1] - mean_y)/std_y # normalization
        dt[:,0] = normalization_x # normalization
        dt[:,1] = normalization_y # normalization
        print(num2)
        if True: # 계속 하나 모자르게 나옴
            video_frame = np.append(video_frame,dt[:,:2])
            
        if np.mean(dt[81:,2])>0.1:
            video_hand_idx.append(int(num2)) # hand frame number
            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())
            
        if num_ls[idx+1] - num_ls[idx] <0 or idx == len(data)-1: # 동영상이 바뀔 때
            hand_frame_ls = hand_frame_ls.reshape(-1,246)
            video_frame = video_frame.reshape(-1,246)
            print(video_frame.shape)
            if num_ls[idx]<max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                
                random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]) # 중복 허용
                random_choice_hand.sort()
            
            for idx,i in enumerate(random_choice_hand):
                video_frame = np.insert(video_frame,i+idx,video_frame[i+idx],axis=0) # 손 부분만 증강
            print(video_frame.shape)
            X_train = np.append(X_train,video_frame).reshape(-1,246)
            video_frame = np.array([])
            hand_frame_ls = np.array([])
#             print(X_train.shape)
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
            
    return X_train
