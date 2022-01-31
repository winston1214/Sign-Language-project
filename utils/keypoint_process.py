import numpy as np
from tqdm.notebook import tqdm
import re
import math

def keti(data): # 전체 frame에 대해서 random하게 증강, sampling
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
    max_frame_num = 50
    n = max_frame_num
    
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

            
        try:
            if num_ls[idx+1]-num_ls[idx] < 0: # 비디오 바뀔 때
                video_frame = np.append(video_frame,dt[:,:2])
#                 hand_frame_ls = hand_frame_ls.reshape(-1,110)
                video_frame = video_frame.reshape(-1,110)
                
                if video_name[idx] == 'KETI_SL_0000041984_173.jpg':
                    avg = np.mean([video_frame[100],video_frame[101]],axis=0)
                    video_frame = np.insert(video_frame,100,avg,axis=0)
                    video_frame = video_frame.reshape(-1,110)
                l = num_ls[idx]
                z = math.floor(l/(n-1))
                y = math.floor((l-z*(n-1))/2) # start point
                r = np.random.randint(1,z+1,n)
                baseline = []

                for i in range(n):
                    baseline.append(y+i*z)
                select_frame = np.array(baseline) + r
                if select_frame[-1] >= len(video_frame):
                    select_frame[-1] = len(video_frame)-1
                video_frame = video_frame[select_frame]
                X_train = np.append(X_train,video_frame).reshape(-1,110)

                video_frame = np.array([])

            else:
                video_frame = np.append(video_frame,dt[:,:2])
        except:
            video_frame = np.append(video_frame,dt[:,:2])
            video_frame = video_frame.reshape(-1,110)
            l = num_ls[idx]
            z = math.floor(l/(n-1))
            y = math.floor((l-z*(n-1))/2) # start point
            r = np.random.randint(1,z,n)
            baseline = []

            for i in range(n):
                baseline.append(y+i*z)
            select_frame = np.array(baseline) + r
            if select_frame[-1] >= len(video_frame):
                select_frame[-1] = len(video_frame)-1
            video_frame = video_frame[select_frame]

            X_train = np.append(X_train,video_frame).reshape(-1,110)

            video_frame = np.array([])

            
    X_train = np.array(X_train).reshape(-1,max_frame_num,110)
    
    return X_train


def hand_normalization(data): # 손 위치 고려
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
#         dt = np.delete(dt,range(13,81),axis=0) # 얼굴 키포인트 제외
        wrist = np.max([dt[9][1], dt[10][1]]) # 허리
        left_hand = np.mean(dt[:,1][81:102])
        right_hand = np.mean(dt[:,1][102:])
        
        if (left_hand < wrist) or (right_hand < wrist): # hand 탐지 될 때
            video_hand_idx.append(int(num2)) # hand frame number
            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())
            
        mean_x,std_x = np.mean(dt[:,0]),np.std(dt[:,0]) # noramlization
        mean_y,std_y = np.mean(dt[:,1]),np.std(dt[:,1]) # normalization
        normalization_x, normalization_y = (dt[:,0] - mean_x)/std_x, (dt[:,1] - mean_y)/std_y # normalization
        dt[:,0] = normalization_x # normalization
        dt[:,1] = normalization_y # normalization  
            
        try:
            if num_ls[idx+1]-num_ls[idx] < 0:
                video_frame = np.append(video_frame,dt[:,:2])
                hand_frame_ls = hand_frame_ls.reshape(-1,246)
                video_frame = video_frame.reshape(-1,246)
                if video_name[idx] == 'KETI_SL_0000041984_173.jpg':
                    avg = np.mean([video_frame[100],video_frame[101]],axis=0)
                    video_frame = np.insert(video_frame,100,avg,axis=0)
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
                else:
                    video_frame = video_frame[-max_frame_num:]

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
            else:
                video_frame = video_frame[-max_frame_num:]
            X_train = np.append(X_train,video_frame).reshape(-1,246)

            video_frame = np.array([])
            hand_frame_ls = np.array([])
            video_hand_idx = []
            
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
    
    return X_train

def face_remove_normalization(data): # 손 위치 고려
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
        wrist = np.max([dt[9][1], dt[10][1]]) # 허리
        left_hand = np.mean(dt[:,1][13:34])
        right_hand = np.mean(dt[:,1][34:])
        
        if (left_hand < wrist) or (right_hand < wrist): # hand 탐지 될 때
            video_hand_idx.append(int(num2)) # hand frame number
            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())
            
        mean_x,std_x = np.mean(dt[:,0]),np.std(dt[:,0]) # noramlization
        mean_y,std_y = np.mean(dt[:,1]),np.std(dt[:,1]) # normalization
        normalization_x, normalization_y = (dt[:,0] - mean_x)/std_x, (dt[:,1] - mean_y)/std_y # normalization
        dt[:,0] = normalization_x # normalization
        dt[:,1] = normalization_y # normalization  
            
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

def all_frame_noraml(data): # original video에서 hand video 부분만 random하게 frame 증강
    np.random.seed(42)
    X_train=np.array([])
    num_ls = []
    video_ls = []
    x_key = []
    y_key = []
    mean_x_ls =[]
    std_x_ls =[]
    mean_y_ls = []
    std_y_ls = []
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
        num1 = i.split('_')[2] # video number
        video_ls.append(int(num1)) # video_ls : video number들의 list
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
    
    # max_frame_num += 1 # 0부터 시작해서
    video_hand_idx = []
    max_frame_num = 376
    print('max_frame_num', max_frame_num)
    video_frame = np.array([])

    for idx,i in tqdm(enumerate(data)): # 평균, 표준편차 list 생성하는 for 문
#         print(idx) # 64
#         print(i) # KETI_SL_0000031451_64.jpg
        # num1 = video_ls[idx] # video name
        try:
            if video_ls[idx+1] - video_ls[idx] <0: # 다음 비디오로 넘어갈때
                dt = data[i]['keypoints']
                dt = np.array(dt).reshape(123,3)
                x_key.append(dt[:,0])
                y_key.append(dt[:,1])
                mean_x,std_x = np.mean(x_key),np.std(x_key) # noramlization value
                mean_y,std_y = np.mean(y_key),np.std(y_key) # normalization
                mean_x_ls.append(mean_x)
                mean_y_ls.append(mean_y)
                std_x_ls.append(std_x)
                std_y_ls.append(std_y)
                x_key = []
                y_key = []
            else: # 계속 같은 비디오인 경우
                dt = data[i]['keypoints']
                dt = np.array(dt).reshape(123,3)
                x_key.append(dt[:,0]) # 한 비디오인 경우 x_keypoint 모으기
                y_key.append(dt[:,1]) # 한 비디오인 경우 y_keypoint 모으기
        except: #마지막 비디오
            dt = data[i]['keypoints']
            dt = np.array(dt).reshape(123,3)
            x_key.append(dt[:,0]) # 한 비디오인 경우 x_keypoint 모으기
            y_key.append(dt[:,1]) # 한 비디오인 경우 y_keypoint 모으기
        
    mean_x,std_x = np.mean(x_key),np.std(x_key) # noramlization value
    mean_y,std_y = np.mean(y_key),np.std(y_key) # normalization
    mean_x_ls.append(mean_x)
    mean_y_ls.append(mean_y)
    std_x_ls.append(std_x)
    std_y_ls.append(std_y)
        # 그럼 지금 mean_x_ls, mean_y_ls, std_x_ls, std_y_ls에는 각 비디오의 평균, 표편이 들어있음
       
        # Run the for loop once more

    j = 0 # video 넘어가는 값 조정
    for idx,i in tqdm(enumerate(data)): # 영상별로 정규화
        try:
            if video_ls[idx+1] - video_ls[idx] <0: # 다음 비디오로 넘어갈때
                dt = data[i]['keypoints']
                dt = np.array(dt).reshape(123,3)
                normalization_x, normalization_y = (dt[:,0] - mean_x_ls[j])/std_x_ls[j], (dt[:,1] - mean_y_ls[j])/std_y_ls[j] # normalization
                dt[:,0] = normalization_x
                dt[:,1] = normalization_y
                j +=1

            else:
                dt = data[i]['keypoints']
                dt = np.array(dt).reshape(123,3)
                normalization_x, normalization_y = (dt[:,0] - mean_x_ls[j])/std_x_ls[j], (dt[:,1] - mean_y_ls[j])/std_y_ls[j] # normalization
                dt[:,0] = normalization_x
                dt[:,1] = normalization_y
        except:
            dt = data[i]['keypoints']
            dt = np.array(dt).reshape(123,3)
            normalization_x, normalization_y = (dt[:,0] - mean_x_ls[j])/std_x_ls[j], (dt[:,1] - mean_y_ls[j])/std_y_ls[j] # normalization
            dt[:,0] = normalization_x
            dt[:,1] = normalization_y

    # print(dt.shape)

    for idx, i in tqdm(enumerate(data)):
        num2 = num_ls[idx]
        dt = data[i]['keypoints']
        dt = np.array(dt).reshape(123,3)

        if np.mean(dt[81:,2])>0.1: # hand 탐지 될 때
            video_hand_idx.append(int(num2)) # hand frame number
            hand_frame_ls = np.append(hand_frame_ls,dt[:,:2].flatten())
            
            
        try:
            if num_ls[idx+1]-num_ls[idx] < 0: # 마지막 frame
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
                    if video_name[idx] == 'KETI_SL_0000041984_173.jpg':
                        avg = np.mean([video_frame[100],video_frame[101]],axis=0)
                        video_frame = np.insert(video_frame,100,avg,axis=0)
                        video_frame = video_frame.reshape(-1,246)
                        print('ignore error')
                    if video_frame.shape[0]> max_frame_num:
                        print(video_frame[:-max_frame_num].shape)
                        video_frame = video_frame[(len(video_frame)-max_frame_num):]
                        print('new:', video_frame.shape)
                X_train = np.append(X_train,video_frame).reshape(-1,246)

                video_frame = np.array([])
                hand_frame_ls = np.array([])
                video_hand_idx = []
            else:
                video_frame = np.append(video_frame,dt[:,:2])

                    
                
        except: # 영상 끝
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
                    
                    if video_frame.shape[0] > max_frame_num:
                        print(video_frame[:-max_frame_num].shape)
                        video_frame = video_frame[(len(video_frame)-max_frame_num):]
                        print('new:', video_frame.shape)
                
            X_train = np.append(X_train,video_frame).reshape(-1,246)

            video_frame = np.array([])
            hand_frame_ls = np.array([])
            video_hand_idx = []
            
    X_train = np.array(X_train).reshape(-1,max_frame_num,246)
    
    return X_train
