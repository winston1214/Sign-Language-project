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
    num_ls = []
    max_frame_num = 0
    min_frame_num = []
    tmp = []
    hand_frame_ls = np.array([])
    video_name = [] # video 밀리는 경우 방지(frame이 0번째부터 있지 않을 수 있음)
    for idx,i in tqdm(enumerate(data)):
        num2 = i.split('_')[3]
        num2 = int(re.sub(".jpg", "", num2))
        num_ls.append(int(num2))
        video_name.append(i)
        try:
            if num_ls[idx+1] - num_ls[idx]<0:
                tmp.append(num2)
                min_frame_num.append([min(tmp)]*len(tmp))
            else:
                tmp.append(num2)
                
        except:
            tmp.append(num2)
            min_frame_num.append([min(tmp)]*len(tmp))
        if max_frame_num < num2: # 최댓값 찾기
            max_frame_num = num2
            
    min_frame_num = sum(min_frame_num,[])
    max_frame_num += 1 # 0부터 시작해서
    video_hand_idx = []
    print('max_frame_num', max_frame_num)
    video_frame = np.array([])

    for idx,i in tqdm(enumerate(data)):
        num2 = num_ls[idx]
        dt = data[i]['keypoints']
        dt = np.array(dt).reshape(123,3)
        mean_x,std_x = np.mean(dt[:,0]),np.std(dt[:,0]) # noramlization
        mean_y,std_y = np.mean(dt[:,1]),np.std(dt[:,1]) # normalization
        normalization_x, normalization_y = (dt[:,0] - mean_x)/std_x, (dt[:,1] - mean_y)/std_y # normalization
        dt[:,0] = normalization_x # normalization
        dt[:,1] = normalization_y # normalization
        
        if np.mean(dt[81:,2])>0.1: # hand 탐지 될 때
            video_hand_idx.append(int(num2)) # hand frame number
            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())
            
            
        try:
            if num_ls[idx+1]-num_ls[idx] < 0:
                video_frame = np.append(video_frame,dt[:,:2])
                hand_frame_ls = hand_frame_ls.reshape(-1,246)
                video_frame = video_frame.reshape(-1,246)


                if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                    if min_frame_num[idx] != 0: # 시작 프레임이 0이 아닐 때
                        start_number = min_frame_num[idx]
                        print(start_number,idx)
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]+start_number-1)
                        for random_idx,h in enumerate(random_choice_hand):
                            insert_num = h+random_idx-start_number
                            video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                    else:
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                        random_choice_hand.sort()
                        for random_idx,h in enumerate(random_choice_hand):

                            video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강

    #                 print(video_frame.shape)
    #                 print(len(random_choice_hand))



    #             print(video_frame.shape)
                if video_frame.shape[0] != max_frame_num: # error check
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
                X_train = np.append(X_train,video_frame).reshape(-1,246)

                video_frame = np.array([])
                hand_frame_ls = np.array([])
                video_hand_idx = []
            else:
                video_frame = np.append(video_frame,dt[:,:2])
        except:
            video_frame = np.append(video_frame,dt[:,:2])
            
            
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
            
    return X_train
