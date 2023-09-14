from PIL import Image
import os
import cv2

def splitFrames(video_, frame_path):
    cap = cv2.VideoCapture(video_)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    num =1
    while True:
        success, data = cap.read()
        if not success:
            break
        data = cv2.resize(data, (320,240))

        cv2.imwrite(frame_path+'/%.6d.jpg' % (num),data)

        num =num+1
    cap.release()

video_path = '/back/dataset/HumanAction/data/NTU/data/videos/nturgbd_rgb_s'
frame_path = '/back/dataset/HumanAction/data/NTU/data/frames/nturgbd_rgb_s'
for i in range(17):
    read_path = video_path+str(i+1)+'/nturgb+d_rgb/'
    save_path = frame_path+str(i+1)+'/nturgb+d_rgb/'

    listfile = os.listdir(read_path)
    for file in listfile:
        videospath =read_path+file
        framespath =save_path+file
        splitFrames(videospath,framespath[:-4])
            
    print(i+1)
    print('done')



