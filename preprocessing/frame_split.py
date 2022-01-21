import subprocess
import shlex
import time
def frame_split(video,output):
    
    command = f'ffmpeg -i {video} {output}%d.jpg -hide_banner -r 30'
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    time.sleep(5)    
    print('video split complete')