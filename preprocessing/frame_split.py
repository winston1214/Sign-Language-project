import imageio
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C:/Users/winst/Downloads/sign_dataset/12995_15508', help='input_folder')
parser.add_argument('--output', type=str, default='C:/Users/winst/Downloads/sign_dataset/12995_15508_frame', help='output_folder')
opt = parser.parse_args()

os.chdir(opt.source)
video = os.listdir(opt.source)
name = list(map(lambda x: x.split('.')[0],video))

for video_name,jpg_name in tqdm(zip(video,name)):
    reader = imageio.get_reader(f'{video_name}')

    for frame_number, im in enumerate(reader):
        # im is numpy array
        imageio.imwrite(f'{opt.output}/{jpg_name}_{frame_number}.jpg', im)