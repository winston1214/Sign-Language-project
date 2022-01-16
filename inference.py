import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from scripts.demo_inference import alphapose_inference
import argparse
import json
from preprocessing.frame_split import frame_split


def inference(opt):
    ### video frame split
    if os.path.exists('frame'):
        indir = 'frame'
    else:
        os.mkdir('frame')
        indir = 'frame'
    frame_split(opt.video,indir)
    alphapose_inference(opt.checkpoint,opt.cfg,opt.format,indir,opt.outdir,opt.sp) # indir로 frame 저장 폴더 만들기

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign-Language-Train')
    parser.add_argument('--checkpoint',type=str,default='pretrained_models/halpe136_fast_res50_256x192.pth',help = 'download pth')
    parser.add_argument('--cfg', type=str, default='configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml',help='cfg')
    parser.add_argument('--format',type=str,default='boaz',help='coco,open,cmu,boaz')
    parser.add_argument('--video',type=str)
    parser.add_argument('--outdir',type=str,default='result/')
    parser.add_argument('--sp', default=True,help = 'if you use multi-gpu, check sp False')
    opt = parser.parse_args()
    inference(opt)