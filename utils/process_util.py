import numpy as np
import math
from scipy.spatial.distance import cosine as cos

## normalization method
def distance_normalization(dt): # Euclidean
    face_r = dt[0] # nose 기준
    body_r = dt[12]
    larm_r = dt[7]
    rarm_r = dt[10]
    lhand_r = dt[13]
    rhand_r = dt[34]
    
    face_c = np.mean(dt[[0,1,2,3,4,11]],axis=0)
    body_c = np.mean(dt[[5,6,12]],axis=0)
    larm_c = np.mean(dt[[7,9]],axis=0)
    rarm_c = np.mean(dt[[8,10]],axis=0)
    lhand_c = np.mean(dt[13:34],axis=0)
    rhand_c = np.mean(dt[34:],axis= 0)
    
    face_d = np.sqrt((face_c[0] - face_r[0])**2 + (face_c[1] - face_r[1])**2)
    body_d = np.sqrt((body_c[0] - body_r[0])**2 + (body_c[1] - body_r[1])**2)
    larm_d = np.sqrt((larm_c[0] - larm_r[0])**2 + (larm_c[1] - larm_r[1])**2)
    rarm_d = np.sqrt((rarm_c[0] - rarm_r[0])**2 + (rarm_c[1] - rarm_r[1])**2)
    lhand_d = np.sqrt((lhand_c[0] - lhand_r[0])**2 + (lhand_c[1] - lhand_r[1])**2)
    rhand_d = np.sqrt((rhand_c[0] - rhand_r[0])**2 + (rhand_c[1] - rhand_r[1])**2)
    new_dt = np.array([])
    for idx,i in enumerate(dt):
        if idx in [0,1,2,3,4,11]:
            normal = [(i[0] - face_c[0])/face_d,(i[1] - face_c[1])/face_d]
        elif idx in [5,6,12]:
            normal = [(i[0] - body_c[0])/body_d,(i[1] - body_c[1])/body_d]
        elif idx in [7,9]:
            normal = [(i[0] - larm_c[0])/larm_d,(i[1] - larm_c[1])/larm_d]
        elif idx in [8,10]:
            normal = [(i[0] - rarm_c[0])/rarm_d,(i[1] - rarm_c[1])/rarm_d]
        elif idx in range(13,34):
            normal = [(i[0] - lhand_c[0])/lhand_d,(i[1] - lhand_c[1])/lhand_d]
        else:
            normal = [(i[0] - rhand_c[0])/rhand_d,(i[1] - rhand_c[1])/rhand_d]
        new_dt = np.append(new_dt,normal)
    return new_dt.reshape(-1,110)
def distance_normalization_mix(dt): # Euclidean + cos
    face_r = dt[0] # nose 기준
    body_r = dt[12]
    larm_r = dt[7]
    rarm_r = dt[10]
    lhand_r = dt[13]
    rhand_r = dt[34]
    
    face_c = np.mean(dt[[0,1,2,3,4,11]],axis=0)
    body_c = np.mean(dt[[5,6,12]],axis=0)
    larm_c = np.mean(dt[[7,9]],axis=0)
    rarm_c = np.mean(dt[[8,10]],axis=0)
    lhand_c = np.mean(dt[13:34],axis=0)
    rhand_c = np.mean(dt[34:],axis= 0)
    distance = []
    for c,r in zip([face_c,body_c,larm_c,rarm_c,lhand_c,rhand_c],[face_r,body_r,larm_r,rarm_r,lhand_r,rhand_r]):
        euc_dis = np.sqrt((c[0]-r[0])**2 + (c[1]-r[1])**2)
        cos_dis = cos(c,r)
        distance.append(euc_dis * 0.5 + cos_dis * 0.5)
    
    new_dt = np.array([])
    for idx,i in enumerate(dt):
        if idx in [0,1,2,3,4,11]:
            normal = [(i[0] - face_c[0])/distance[0],(i[1] - face_c[1])/distance[0]]
        elif idx in [5,6,12]:
            normal = [(i[0] - body_c[0])/distance[1],(i[1] - body_c[1])/distance[1]]
        elif idx in [7,9]:
            normal = [(i[0] - larm_c[0])/distance[2],(i[1] - larm_c[1])/distance[2]]
        elif idx in [8,10]:
            normal = [(i[0] - rarm_c[0])/distance[3],(i[1] - rarm_c[1])/distance[3]]
        elif idx in range(13,34):
            normal = [(i[0] - lhand_c[0])/distance[4],(i[1] - lhand_c[1])/distance[4]]
        else:
            normal = [(i[0] - rhand_c[0])/distance[5],(i[1] - rhand_c[1])/distance[5]]
        new_dt = np.append(new_dt,normal)
    return new_dt.reshape(-1,110)
    
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

def dynamic_probability(n,p_n):
    down = [i for i in range(p_n,1,-1)] + [i for i in range(3,p_n+1)]
    up = [1] * (p_n-1) + [i for i in range(2,p_n)]
    p = np.array(up)/np.array(down)
    prob = np.array([0]*n,dtype=np.float64)
    for i in p:
        prob += binomial_prob(n-1,i)
    make_prob = prob/len(p)
    final_prob = sorted(make_prob[:n//2]) + sorted(make_prob[n//2:],reverse=True)
    return final_prob