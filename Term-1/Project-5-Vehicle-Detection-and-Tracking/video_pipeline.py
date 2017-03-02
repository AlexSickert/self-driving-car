
import training as trn
from moviepy.editor import *

what = 2

if what == 1:
    video_path = "./video/test_video.mp4"
    white_output = './video/test_video_output.mp4'

if what == 2:
    video_path = "./video/project_video.mp4"
    white_output = './video/project_video_output.mp4'

print(video_path)
clip1 = VideoFileClip(video_path)
white_clip = clip1.fl_image(trn.process_video) 
white_clip.write_videofile(white_output, audio=False)  