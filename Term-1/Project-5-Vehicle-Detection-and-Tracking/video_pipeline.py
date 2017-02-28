
import training as trn
from moviepy.editor import *

video_path = "./video/test_video.mp4"
white_output = './video/test_video_output.mp4'


clip1 = VideoFileClip(video_path)
white_clip = clip1.fl_image(trn.process_image) 
white_clip.write_videofile(white_output, audio=False)  