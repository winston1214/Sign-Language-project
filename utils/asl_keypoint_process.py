import os
import numpy as np
import json
from tqdm.notebook import tqdm
import math
import re
def combinations_len(n,r):
    return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))
def binomial_prob(n,p):
    prob = []
    for r in range(n+1):
        prob.append(combinations_len(n,r) * (p**r) * ((1-p)**(n-r)))
    return prob
def probability(n):
    half_prob = binomial_prob(n-1,1/2)
    qun_prob = binomial_prob(n-1,1/3)
    qun2_prob = binomial_prob(n-1,2/3)
    make_prob = np.array(half_prob) + np.array(qun_prob) + np.array(qun2_prob)
    make_prob /= 3
    final_prob = sorted(make_prob[:n//2]) + sorted(make_prob[n//2:],reverse=True)
    return final_prob


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

import os
def asl_skip_aug_prob_sort2(path,max_frame_num=116):
    if path[-1] != '/':
        path += '/'
    json_ls = os.listdir(path)
    X_data = np.array([])
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

        final_prob = probability(len(video_key))
        if len(video_key) < n:
            insert_num = n - len(video_key)
            if len(video_key) < insert_num:
                random_choice_hand = np.random.choice(range(len(video_key)),len(video_key),p = final_prob,replace=False)
                while (len(video_key) + len(random_choice_hand)) != n:
                    present = len(video_key) + len(random_choice_hand)
                    
                    if n - present >= len(video_key):

                        random_choice_hand = np.append(random_choice_hand,np.random.choice(range(len(video_key)),len(video_key),p = final_prob,replace=False))
                    else:
                        random_choice_hand = np.append(random_choice_hand,np.random.choice(range(len(video_key)),n - present,p = final_prob,replace=False))

            else:
                random_choice_hand = np.random.choice(range(len(video_key)),n - len(video_key),p = final_prob,replace=False)

            random_choice_hand.sort()                      

            for random_idx,h in enumerate(random_choice_hand):
                insert_num = h+random_idx-1
                video_key = np.insert(video_key,insert_num,video_key[insert_num],axis=0) # 손 부분만 증강
        
            
        else: # keti

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
    return X_data

import os
def asl_all_frame_aug(path,max_frame_num=116):
    if path[-1] != '/':
        path += '/'
    json_ls = os.listdir(path)
    X_data = np.array([])
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
        
        select_frame = np.random.choice(range(len(video_key)),max_frame_num - len(video_key))

        select_frame.sort()
        for random_idx,h in enumerate(select_frame):
            video_key = np.insert(video_key,h+random_idx,video_key[h+random_idx],axis=0) # 손 부분만 증강

        X_data = np.append(X_data,video_key)
    X_data = X_data.reshape(-1,n,110)
    return X_data
def asl_prob_sampling(path,max_frame_num=116):
    if path[-1] != '/':
        path += '/'
    json_ls = os.listdir(path)
    X_data = np.array([])
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
        final_prob = probability(len(video_key))
        select_frame = np.random.choice(range(len(video_key)),max_frame_num,p = final_prob,replace = False)
        
        select_frame.sort()
        video_key = video_key[select_frame]
        X_data = np.append(X_data,video_key)
    X_data = X_data.reshape(-1,n,110)
    return X_data