import cv2
import os
import copy
import numpy as np
import argparse
import json
from collections import OrderedDict
from src import model
from src import util
from src.body import Body
from src.hand import Hand
from pathlib import Path

file_data = OrderedDict()

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
parser.add_argument('--source', type=str, default='/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994_frame', help='input_folder')
parser.add_argument('--save_dir',type='str',default = '/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994_frame_openpose/',help='save folder')
args = parser.parse_args()

path = args.source
save_dir = args.save_dir
if save_dir[-1] != '/':
    save_dir = save_dir + '/'
pathlist = Path('/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994_frame').glob('**/*.jpg')

for test_image in pathlist:
    os.chdir(path)
    oriImg = cv2.imread(test_image.name)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks
        #
        #   [:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)


    file_data["original_frame"] = test_image.name

    if len(candidate)==0:
        continue
    else:
        file_data["candidate"] = candidate.tolist()
        file_data["subset"] = subset.tolist()

    if len(all_hand_peaks)==0:
        file_data["left_hand_key_point"] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],[0, 0],
                                            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0,0],[0,0]]
        file_data["right_hand_key_point"] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],[0, 0],
                                            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0,0],[0,0]]
    else:
        file_data["left_hand_key_point"] = all_hand_peaks[0].tolist()
        file_data["right_hand_key_point"] = all_hand_peaks[1].tolist()

    with open(save_dir+test_image.name.rstrip('.jpg')+'_keypoint.json','w',encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")

