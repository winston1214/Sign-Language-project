import imageio
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994', help='input_folder')
parser.add_argument('--output_path', type=str, default='/content/drive/MyDrive/BOAZ_수어프로젝트/Data/10481_12994_frame', help='output_folder')
opt = parser.parse_args()

os.chdir(opt.source)
video = os.listdir(opt.source)
name = list(map(lambda x: x.split('.')[0],video))

for video_name,jpg_name in tqdm(zip(video,name)):
    reader = imageio.get_reader(f'{video_name}')

    for frame_number, im in enumerate(reader):
        # im is numpy array
        imageio.imwrite(f'{jpg_name}_{frame_number}.jpg', im)