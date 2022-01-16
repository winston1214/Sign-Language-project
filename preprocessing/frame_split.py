import subprocess
import shlex
def frame_split(video,output):
    video_name = video.split('.')[0]
    command = f'ffmpeg -i {video} {output}/{video_name}%d.jpg -hide_banner -r 30'
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print('video split complete')