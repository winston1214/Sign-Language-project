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
    video_set = set()
    hand_frame_ls = np.array([])
    minimum = 1e+5
    video_dic = {}
    video_name = [] # video 밀리는 경우 방지(frame이 0번째부터 있지 않을 수 있음)
    for idx,i in tqdm(enumerate(data)):
        # print(idx) # 반복 횟수 1
        # print(i) # video name : KETI_SL_0000035402_0.jpg
        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))
         # print(num2) # frame number
        num_ls.append(int(num2))   # num_ls : frame number들의 list   num_ls = [0]
        video_name.append(i) # video name : KETI_SL_0000035402_90.jpg
        video_set.add('_'.join(i.split('.')[0].split('_')[:-1]))

    for idx,i in tqdm(enumerate(data)):

        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))

        if max_frame_num < num2: # 최댓값 frame 찾기
            max_frame_num = num2
            
        try: 
            if num_ls[idx+1] - num_ls[idx] <0: # num_ls : frame number들의 list # 123->1로 frame number가 넘어갈때, 1<123이므로 1-123<0
                min_frame_num.append(minimum) 
                minimum = 1e+5
            else:
                if minimum > num2:
                    minimum = num2
        except: # 끝
            if minimum > num2:
                minimum = num2
            min_frame_num.append(minimum)
    video_set = sorted(list(video_set))
    for k,v in zip(video_set,min_frame_num):
        video_dic[k] = v

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
                    start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                    if len(video_hand_idx) == 0: # 손이 하나도 안뽑힐 때
                        video_hand_idx = list(range(num_ls[idx]))
                    if start_number != 0: # 시작 프레임이 0이 아닐 때
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]+start_number-1)
                        
                        for random_idx,h in enumerate(random_choice_hand):
                            insert_num = h+random_idx-start_number
                            video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                    else:
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                        random_choice_hand.sort()
                        for random_idx,h in enumerate(random_choice_hand):

                            video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
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

            hand_frame_ls = hand_frame_ls.reshape(-1,246)
            video_frame = video_frame.reshape(-1,246)

            if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                if len(video_hand_idx) == 0:
                    video_hand_idx = list(range(num_ls[idx]))
                if start_number != 0: # 시작 프레임이 0이 아닐 때
                    
                    random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]+start_number-1)

                    for random_idx,h in enumerate(random_choice_hand):
                        insert_num = h+random_idx-start_number
                        video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                else:
                    random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                    random_choice_hand.sort()
                    for random_idx,h in enumerate(random_choice_hand):

                        video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
            X_train = np.append(X_train,video_frame).reshape(-1,246)

            video_frame = np.array([])
            hand_frame_ls = np.array([])
            video_hand_idx = []
            
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
    
    return X_train

def body_hand_normalization(data): # face 제외하고 hand 부분 random 하게 추출
    np.random.seed(42)
    X_train=np.array([])
    num_ls = []
    max_frame_num = 0
    min_frame_num = []
    video_set = set()
    hand_frame_ls = np.array([])
    minimum = 1e+5
    video_dic = {}
    video_name = [] # video 밀리는 경우 방지(frame이 0번째부터 있지 않을 수 있음)
    for idx,i in tqdm(enumerate(data)):
        # print(idx) # 반복 횟수 1
        # print(i) # video name : KETI_SL_0000035402_0.jpg
        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))
         # print(num2) # frame number
        num_ls.append(int(num2))   # num_ls : frame number들의 list   num_ls = [0]
        video_name.append(i) # video name : KETI_SL_0000035402_90.jpg
        video_set.add('_'.join(i.split('.')[0].split('_')[:-1]))

    for idx,i in tqdm(enumerate(data)):

        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))

        if max_frame_num < num2: # 최댓값 frame 찾기
            max_frame_num = num2
            
        try: 
            if num_ls[idx+1] - num_ls[idx] <0: # num_ls : frame number들의 list # 123->1로 frame number가 넘어갈때, 1<123이므로 1-123<0
                min_frame_num.append(minimum) 
                minimum = 1e+5
            else:
                if minimum > num2:
                    minimum = num2
        except: # 끝
            if minimum > num2:
                minimum = num2
            min_frame_num.append(minimum)
    video_set = sorted(list(video_set))
    for k,v in zip(video_set,min_frame_num):
        video_dic[k] = v

    max_frame_num += 1 # 0부터 시작해서
    video_hand_idx = []
    print('max_frame_num', max_frame_num)
    video_frame = np.array([])

    for idx,i in tqdm(enumerate(data)):
        num2 = num_ls[idx]
        dt = data[i]['keypoints']
        dt = np.array(dt).reshape(123,3)
        dt = np.delete(dt,range(13,81),axis=0) # 얼굴 키포인트 제외
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
                hand_frame_ls = hand_frame_ls.reshape(-1,110)
                video_frame = video_frame.reshape(-1,110)
                if video_name[idx] == 'KETI_SL_0000041984_173.jpg':
                    avg = np.mean([video_frame[100],video_frame[101]],axis=0)
                    video_frame = np.insert(video_frame,100,avg,axis=0)
                    video_frame = video_frame.reshape(-1,110)

                if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                    start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                    if len(video_hand_idx) == 0: # 손이 하나도 안뽑힐 때
                        video_hand_idx = list(range(num_ls[idx]))
                    if start_number != 0: # 시작 프레임이 0이 아닐 때
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]+start_number-1)
                        
                        for random_idx,h in enumerate(random_choice_hand):
                            insert_num = h+random_idx-start_number
                            video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                    else:
                        random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                        random_choice_hand.sort()
                        for random_idx,h in enumerate(random_choice_hand):

                            video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강
                else:
                    video_frame = video_frame[-max_frame_num:]

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
                X_train = np.append(X_train,video_frame).reshape(-1,110)

                video_frame = np.array([])
                hand_frame_ls = np.array([])
                video_hand_idx = []
            else:
                video_frame = np.append(video_frame,dt[:,:2])
        except:
            video_frame = np.append(video_frame,dt[:,:2])

            hand_frame_ls = hand_frame_ls.reshape(-1,110)
            video_frame = video_frame.reshape(-1,110)

            if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                if len(video_hand_idx) == 0:
                    video_hand_idx = list(range(num_ls[idx]))
                if start_number != 0: # 시작 프레임이 0이 아닐 때
                    
                    random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]+start_number-1)

                    for random_idx,h in enumerate(random_choice_hand):
                        insert_num = h+random_idx-start_number
                        video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                else:
                    random_choice_hand = np.random.choice(video_hand_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                    random_choice_hand.sort()
                    for random_idx,h in enumerate(random_choice_hand):

                        video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
            else:
                video_frame = video_frame[-max_frame_num:]
            X_train = np.append(X_train,video_frame).reshape(-1,110)

            video_frame = np.array([])
            hand_frame_ls = np.array([])
            video_hand_idx = []
            
    X_train = np.array(X_train).reshape(-1,max_frame_num,110)
    
    return X_train

def all_frame_random(data): # 전체 frame에 대해서 random하게 증강
    np.random.seed(42)
    X_train=np.array([])
    num_ls = []
    max_frame_num = 0
    min_frame_num = []
    video_set = set()
    minimum = 1e+5
    video_dic = {}
    video_name = [] # video 밀리는 경우 방지(frame이 0번째부터 있지 않을 수 있음)
    for idx,i in tqdm(enumerate(data)):
        # print(idx) # 반복 횟수 1
        # print(i) # video name : KETI_SL_0000035402_0.jpg
        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))
         # print(num2) # frame number
        num_ls.append(int(num2))   # num_ls : frame number들의 list   num_ls = [0]
        video_name.append(i) # video name : KETI_SL_0000035402_90.jpg
        video_set.add('_'.join(i.split('.')[0].split('_')[:-1]))

    for idx,i in tqdm(enumerate(data)):

        num2 = i.split('_')[3] # 0
        num2 = int(re.sub(".jpg", "", num2))

        if max_frame_num < num2: # 최댓값 frame 찾기
            max_frame_num = num2
            
        try: 
            if num_ls[idx+1] - num_ls[idx] <0: # num_ls : frame number들의 list # 123->1로 frame number가 넘어갈때, 1<123이므로 1-123<0
                min_frame_num.append(minimum) 
                minimum = 1e+5
            else:
                if minimum > num2:
                    minimum = num2
        except: # 끝
            if minimum > num2:
                minimum = num2
            min_frame_num.append(minimum)
    video_set = sorted(list(video_set))
    for k,v in zip(video_set,min_frame_num):
        video_dic[k] = v

    max_frame_num += 1 # 0부터 시작해서

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
        
        
            
            
        try:
            if num_ls[idx+1]-num_ls[idx] < 0:
                video_frame = np.append(video_frame,dt[:,:2])
#                 hand_frame_ls = hand_frame_ls.reshape(-1,246)
                video_frame = video_frame.reshape(-1,246)
                
                if video_name[idx] == 'KETI_SL_0000041984_173.jpg':
                    avg = np.mean([video_frame[100],video_frame[101]],axis=0)
                    video_frame = np.insert(video_frame,100,avg,axis=0)
                    video_frame = video_frame.reshape(-1,246)
                    
                video_frame_idx = [i for i in range(len(video_frame))] # 한 비디오에 대한 비디오 프레임 인덱스
                
                if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                    start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                    if start_number != 0: # 시작 프레임이 0이 아닐 때
                        random_choice_hand = np.random.choice(video_frame_idx,max_frame_num - num_ls[idx]+start_number-1)
                        
                        for random_idx,h in enumerate(random_choice_hand):
                            insert_num = h+random_idx-start_number
                            video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                    else:
                        random_choice_hand = np.random.choice(video_frame_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                        random_choice_hand.sort()
                        for random_idx,h in enumerate(random_choice_hand):

                            video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강
                else:
                    video_frame = video_frame[-max_frame_num:]

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
                X_train = np.append(X_train,video_frame).reshape(-1,246)

                video_frame = np.array([])

            else:
                video_frame = np.append(video_frame,dt[:,:2])
        except:
            video_frame = np.append(video_frame,dt[:,:2])


            video_frame = video_frame.reshape(-1,246)

            if video_frame.shape[0] < max_frame_num: # 비디오프레임이 최대 비디오프레임보다 적을 때
                start_number = video_dic['_'.join(video_name[idx].split('.')[0].split('_')[:-1])]
                video_frame_idx = [i for i in range(len(video_frame))]
                if start_number != 0: # 시작 프레임이 0이 아닐 때
                    
                    random_choice_hand = np.random.choice(video_frame_idx,max_frame_num - num_ls[idx]+start_number-1)

                    for random_idx,h in enumerate(random_choice_hand):
                        insert_num = h+random_idx-start_number
                        video_frame = np.insert(video_frame,insert_num,video_frame[insert_num],axis=0) # 손 부분만 증강
                else:
                    random_choice_hand = np.random.choice(video_frame_idx,max_frame_num - num_ls[idx]-1) # 중복 허용
                    random_choice_hand.sort()
                    for random_idx,h in enumerate(random_choice_hand):

                        video_frame = np.insert(video_frame,h+random_idx,video_frame[h+random_idx],axis=0) # 손 부분만 증강

                if video_frame.shape[0] != max_frame_num: # error check
                    print('error!')
                    print(list(data.keys())[idx]) # error check
                    print(video_frame.shape) # error check
            else:
                video_frame = video_frame[-max_frame_num:]
            X_train = np.append(X_train,video_frame).reshape(-1,246)

            video_frame = np.array([])

            
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
    
    return X_train
