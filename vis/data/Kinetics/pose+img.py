
from PIL import Image
import os
import cv2
from cv2 import waitKey
import numpy as np
import math 
import random
import os.path as osp
import json



def load_img(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    return img

def load_video(path, frame_num):
    
    # load frames
    video = []
    pose_frame_idxs=[]
    for i in range(frame_num):
    
        img = load_img(osp.join(path, '%.6d.jpg' % (i+1))) # 1-based
        video.append(img)
        pose_frame_idxs.append(i)

    # frame indexs for pose sampling
    return video, pose_frame_idxs

def load_skeleton(skeleton_path, frame_idxs, resized_shape):
    with open(skeleton_path) as f:
        skeleton_data = json.load(f)
    pose_outs = skeleton_data['data']
    
    pose_coords = np.ones((len(frame_idxs), top_k_pose, joint_num, 2), dtype=np.float32)
    pose_scores = np.zeros((len(frame_idxs), top_k_pose, joint_num), dtype=np.float32)
    for pose_out in pose_outs:
        skeleton_out, frame_idx = pose_out['skeleton'], pose_out['frame_index']
        if frame_idx not in frame_idxs:
            continue
        
        person_num = len(skeleton_out)
        pose_coords_per_frame = np.ones((person_num, joint_num, 2), dtype=np.float32)
        pose_scores_per_frame = np.zeros((person_num, joint_num), dtype=np.float32)
        for pid in range(person_num):
            pose_coord, pose_score = skeleton_out[pid]['pose'], skeleton_out[pid]['score']
            pose_coord = np.array(pose_coord).reshape(joint_num,2)
            pose_score = np.array(pose_score).reshape(joint_num)
            
            # resize from (0~1) to resized_shape
            pose_coord[:,0] = pose_coord[:,0] * resized_shape[1]
            pose_coord[:,1] = pose_coord[:,1] * resized_shape[0]

            pose_coords_per_frame[pid] = pose_coord
            pose_scores_per_frame[pid] = pose_score
        
        # select top-k pose
        if person_num < top_k_pose:
            pose_coords_per_frame = np.concatenate((pose_coords_per_frame, np.ones((top_k_pose - person_num, joint_num, 2))))
            pose_scores_per_frame = np.concatenate((pose_scores_per_frame, np.zeros((top_k_pose - person_num, joint_num))))
        top_k_idx = np.argsort(np.mean(pose_scores_per_frame,1))[-top_k_pose:][::-1]
        pose_coords_per_frame = pose_coords_per_frame[top_k_idx,:,:]
        pose_scores_per_frame = pose_scores_per_frame[top_k_idx,:]
        
        idx = frame_idxs.index(frame_idx)
        pose_coords[idx] = pose_coords_per_frame
        pose_scores[idx] = pose_scores_per_frame
    
    return pose_coords, pose_scores

def add_joints(image, joints):
    coco_part_labels = ['Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3']

    coco_part_orders = [ (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (0,14), (0,15), (14,16), (15,17)]

    part_orders = coco_part_orders

    def link(a, b):
        if a < joints.shape[0] and b < joints.shape[0]:
            jointa = joints[a]
            jointb = joints[b]
            cv2.line(image, (int(jointa[0]), int(jointa[1])), (int(jointb[0]), int(jointb[1])), (255, 255, 255), 3)

    # add joints
    for joint in joints:
        cv2.circle(image, (int(joint[0]), int(joint[1])), 3, (0, 255, 255), 7)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1])

    return image

if __name__ =="__main__":


    skeleton_path =  "/home/zj-user/Icode/vis/data/Kinetics/skeleton/-3oo45vpQV4.json"
    img_path = "/home/zj-user/Icode/vis/data/Kinetics/frames/writing/-3oo45vpQV4"
    save_path = "/home/zj-user/Icode/vis/data/Kinetics/figs/-3oo45vpQV4"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_size_h_w =(256,454)
    img_size_w_h =(454,256)
    frame_num = len(os.listdir(img_path))
    top_k_pose=5

    class_num = 50
    joint_num = 18
    joint_names = ('Nose', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear') # openpose joint set
    flip_pairs = ( (2,5), (3,6), (4,7), (8,11), (9,12), (10,13), (14,15), (16,17) )
    skeleton = ( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )

    video_name = osp.join(save_path,'pose+img.mp4')
    fourcc =cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(video_name,fourcc, 30.0,img_size_w_h,True)

    video, frame_idxs =load_video(img_path,frame_num)
    
    pose_coords, pose_scores = load_skeleton(skeleton_path,frame_idxs, img_size_h_w)

    for i in range(frame_num):
        
        img_path_i =osp.join(save_path, '%.6d.jpg' % (i+1))
        pose_coords_per_frame = pose_coords[i]
        person_num =pose_coords_per_frame.shape[0]
       
       
        for j in range(person_num):
            
            img = add_joints(video[i],pose_coords_per_frame[j])
        
        videowrite.write(img)

        cv2.imwrite(img_path_i,img)
            
        # cv2.imshow("image",img)
        # waitKey(100)



