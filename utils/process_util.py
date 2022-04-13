import numpy as np
import math
from scipy.spatial.distance import cosine as cos
from sklearn.preprocessing import MinMaxScaler

def robust_normal_minmax(dt): # Euclidean
    face_r = dt[0] # nose 기준
    body_r = dt[12]
    larm_r = dt[7]
    rarm_r = dt[10]
    # lhand_r = dt[13]
    # rhand_r = dt[34]

    c_x,c_y = np.mean(dt[:,0]),np.mean(dt[:,1])
    
    face_d = np.sqrt((c_x - face_r[0])**2 + (c_y - face_r[1])**2)
    body_d = np.sqrt((c_x - body_r[0])**2 + (c_y - body_r[1])**2)
    larm_d = np.sqrt((c_x - larm_r[0])**2 + (c_y - larm_r[1])**2)
    rarm_d = np.sqrt((c_x - rarm_r[0])**2 + (c_y - rarm_r[1])**2)
    # lhand_d = np.sqrt((c_x - lhand_r[0])**2 + (c_y - lhand_r[1])**2)
    # rhand_d = np.sqrt((c_x - rhand_r[0])**2 + (c_y - rhand_r[1])**2)
    for idx,i in enumerate(dt):
        if idx in [0,1,2,3,4,11]:
            dt[idx] = [(i[0] - c_x)/face_d,(i[1] - c_y)/face_d]
        elif idx in [5,6,12]:
            dt[idx] = [(i[0] - c_x)/body_d,(i[1] - c_y)/body_d]
        elif idx in [7,9]:
            dt[idx] = [(i[0] - c_x)/larm_d,(i[1] - c_y)/larm_d]
        elif idx in [8,10]:
            dt[idx] = [(i[0] - c_x)/rarm_d,(i[1] - c_y)/rarm_d]
    scaler = MinMaxScaler()
    
    r_x = scaler.fit_transform(dt[13:34][:,0].reshape(-1,1))-0.5
    r_y = scaler.fit_transform(dt[13:34][:,1].reshape(-1,1))-0.5
    l_x = scaler.fit_transform(dt[34:][:,0].reshape(-1,1))-0.5
    l_y = scaler.fit_transform(dt[34:][:,1].reshape(-1,1))-0.5
    dt[13:34][:,0] =  r_x.reshape(-1)
    dt[13:34][:,1] = r_y.reshape(-1)
    dt[34:][:,0] = l_x.reshape(-1)
    dt[34:][:,1] = l_y.reshape(-1)

    
    return dt.reshape(55,-1)
def distance_normalization(dt): # Euclidean
    face_r = dt[0] # nose 기준
    body_r = dt[12]
    larm_r = dt[7]
    rarm_r = dt[10]
    lhand_r = dt[13]
    rhand_r = dt[34]

    c_x,c_y = np.mean(dt[:,0]),np.mean(dt[:,1])
    
    face_d = np.sqrt((c_x - face_r[0])**2 + (c_y - face_r[1])**2)
    body_d = np.sqrt((c_x - body_r[0])**2 + (c_y - body_r[1])**2)
    larm_d = np.sqrt((c_x - larm_r[0])**2 + (c_y - larm_r[1])**2)
    rarm_d = np.sqrt((c_x - rarm_r[0])**2 + (c_y - rarm_r[1])**2)
    lhand_d = np.sqrt((c_x - lhand_r[0])**2 + (c_y - lhand_r[1])**2)
    rhand_d = np.sqrt((c_x - rhand_r[0])**2 + (c_y - rhand_r[1])**2)
    new_dt = np.array([])
    for idx,i in enumerate(dt):
        if idx in [0,1,2,3,4,11]:
            normal = [(i[0] - c_x)/face_d,(i[1] - c_y)/face_d]
        elif idx in [5,6,12]:
            normal = [(i[0] - c_x)/body_d,(i[1] - c_y)/body_d]
        elif idx in [7,9]:
            normal = [(i[0] - c_x)/larm_d,(i[1] - c_y)/larm_d]
        elif idx in [8,10]:
            normal = [(i[0] - c_x)/rarm_d,(i[1] - c_y)/rarm_d]
        elif idx in range(13,34):
            normal = [(i[0] - c_x)/lhand_d,(i[1] - c_y)/lhand_d]
        else:
            normal = [(i[0] - c_x)/rhand_d,(i[1] - c_y)/rhand_d]
        new_dt = np.append(new_dt,normal)
    return new_dt.reshape(55,-1)

def distance_normalization_right_shoulder(dt): # Euclidean
    right_s = dt[6]
    c_x,c_y = np.mean(dt[:,0]),np.mean(dt[:,1])
    d = np.sqrt((c_x - right_s[0])**2 + (c_y - right_s[1])**2)
    new_dt = np.array([])
    for i in dt:
        normal = [(i[0] - c_x)/d,(i[1] - c_y)/d]
        new_dt = np.append(new_dt,normal)
    return new_dt.reshape(55,-1)

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