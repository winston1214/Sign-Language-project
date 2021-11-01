import skvideo.io  
import argparse
import os
from tqdm import tqdm

# import easydict
 
# args = easydict.EasyDict(
#     {
#         "source" : '/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994' ,
#         "output_path" : '/content/drive/MyDrive/BOAZ_수어프로젝트/Data/frame'
#     }

# )
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C:/Users/winst/Desktop/sign_language/preprocessing/video', help='input_folder')
parser.add_argument('--output', type=str, default='C:/Users/winst/Desktop/sign_language/preprocessing/jpg', help='output_folder')
opt = parser.parse_args()

os.chdir(opt.source)
video = os.listdir(opt.source)
name = list(map(lambda x: x.split('.')[0],video))

for video_name,jpg_name in tqdm(zip(video,name)):
    reader = skvideo.io.vread(f'{video_name}')

    for frame_number, im in enumerate(reader):
        # im is numpy array
        skvideo.io.vwrite(f'{opt.output}/{jpg_name}_{frame_number}.jpg', im)