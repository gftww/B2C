import shutil
from PIL import Image
import os
import cv2


def splitFrames(video_path):
    cap = cv2.VideoCapture(video_path+'.avi')

    if not os.path.exists(video_path):
            os.makedirs(video_path)
    
    num =1
    while True:
        success, data = cap.read()
        if not success:
            break

        cv2.imwrite(video_path+'/%.6d.jpg' % (num),data)

        num =num+1

    cap.release()

def skeletoncopyfile(srcfile, dstpath= '/home/gft/PycharmProjects/vis/data/NTU/samples'):
    source_path =os.path.join('/home/gft/PycharmProjects/vis/data/NTU/data/nturgb+d_skeletons',(srcfile[:-4]+'.skeleton')) 

    if not os.path.isfile(source_path):
        print("%s not exist"%(source_path))
    else:
        shutil.copy(source_path,dstpath)
        print("copy %s -> %s"%(source_path,dstpath))

    

base_path = '/home/gft/PycharmProjects/vis/data/NTU/samples'
listfile = os.listdir(base_path)
for file in listfile:

    if (file[-4:]=='.avi'):

        filepath = base_path+'/'+file
        splitFrames(filepath[:-4])
        skeletoncopyfile(file[:-4],base_path)



