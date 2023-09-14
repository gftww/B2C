
from PIL import Image
import os
import cv2
from cv2 import waitKey
import numpy as np
import math 
import random
import os.path as osp




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

def load_skeleton(skeleton_path, frame_idxs, original_shape, resized_shape):

    ''''
    skeleton_path:  /path/to/****.skeleton, 
    frame_idxs: a list of sampled dataset
    original_shape:(img_height, img_width)
    resized_shape:(img_height, img_width)
    '''
    with open(skeleton_path) as f:
        skeleton_data = f.readlines()

    line_idx = 0
    cur_fid = 0
    pose_coords = np.ones((len(frame_idxs), top_k_pose,joint_num, 2), dtype=np.float32) # joint coordinates from all frames in frame_idxs
    pose_scores = np.zeros((len(frame_idxs), top_k_pose,joint_num), dtype=np.float32)
    while cur_fid <= frame_idxs[-1]:
        line_idx += 1
        person_num = int(skeleton_data[line_idx])

        if cur_fid in frame_idxs:
            pose_coords_per_frame = np.ones((person_num,joint_num, 2), dtype=np.float32) # joint coordinates per frame
            pose_scores_per_frame = np.zeros((person_num,joint_num), dtype=np.float32)
            for pid in range(person_num):
                line_idx += 2
                pose_coord = np.ones((joint_num,2), dtype=np.float32)
                pose_score = np.ones((joint_num), dtype=np.float32)
                for j in range(joint_num):
                    line_idx += 1
                    #25 lines for each joint,the sixth and seventh number are x and y 
                    pose_coord[j][0] = float(skeleton_data[line_idx].split()[5])
                    pose_coord[j][1] = float(skeleton_data[line_idx].split()[6])
                    
                    # There are some NaN joint coordinates in skeleton annotation files
                    if math.isnan(pose_coord[j][0]) or math.isnan(pose_coord[j][1]):
                        pose_coord[j] = 0
                        pose_score[j] = 0

                # resize to video_shape
                pose_coord[:,0] = pose_coord[:,0] / original_shape[1] * resized_shape[1]
                pose_coord[:,1] = pose_coord[:,1] / original_shape[0] * resized_shape[0]

                pose_coords_per_frame[pid] = pose_coord
                pose_scores_per_frame[pid] = pose_score
            
            # select top-k pose
            if person_num < top_k_pose:
                pose_coords_per_frame = np.concatenate((pose_coords_per_frame, np.ones((top_k_pose - person_num,joint_num, 2))))
                pose_scores_per_frame = np.concatenate((pose_scores_per_frame, np.zeros((top_k_pose - person_num,joint_num))))
            top_k_idx = np.argsort(np.mean(pose_scores_per_frame,1))[-top_k_pose:][::-1]
            pose_coords_per_frame = pose_coords_per_frame[top_k_idx,:,:]
            pose_scores_per_frame = pose_scores_per_frame[top_k_idx,:]
            
            idx = frame_idxs.index(cur_fid)
            pose_coords[idx] = pose_coords_per_frame # save pose_coord of cur_fid frame
            pose_scores[idx] = pose_scores_per_frame
            cur_fid += 1
        else:
            line_idx += person_num * (2 +joint_num )
            cur_fid += 1
    
    return pose_coords, pose_scores

def add_joints(image, joints, c=True):
    coco_part_labels = ['Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3']

    coco_part_orders = [ (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20)]

    part_orders = coco_part_orders

    def link(a, b):
        if a < joints.shape[0] and b < joints.shape[0]:
            jointa = joints[a]
            jointb = joints[b]
            if c :
                cv2.line(image, (int(jointa[0]), int(jointa[1])), (int(jointb[0]), int(jointb[1])), (255, 255, 255), 3)
            else:
                cv2.line(image, (int(jointa[0]), int(jointa[1])), (int(jointb[0]), int(jointb[1])), (0,125, 255), 3)

    # add joints
    for joint in joints:
        if c:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 3, (0, 255, 255), 7)
        else:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), 7)
    # add link
    for pair in part_orders:
        link(pair[0], pair[1])

    return image

if __name__ =="__main__":


    skeleton_path =  "/home/gft/PycharmProjects/vis/data/NTU/samples/S005C001P013R002A047.skeleton"
    img_path = "/home/gft/PycharmProjects/vis/data/NTU/samples/S005C001P013R002A047_rgb"
    save_path = "/home/gft/PycharmProjects/vis/data/NTU/figs/S005C001P013R002A047/pose_seq"

    if not os.path.exists(save_path):
         os.makedirs(save_path)

    img_size_h_w =(1080,1920)
    img_size_w_h =(1920,1080)
    frame_num = len(os.listdir(img_path))
    top_k_pose=5

    class_num = 60
    joint_num = 25
    joint_names = ('Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3')
    flip_pairs = ( (4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18), (15,19), (21,23), (22,24) )
    skeleton = ( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )


    video, frame_idxs =load_video(img_path,frame_num)
    
    pose_coords, pose_scores = load_skeleton(skeleton_path,frame_idxs, img_size_h_w , img_size_h_w )

    i =21
    j =22

    img_path = osp.join(save_path, 'img_%.6d.jpg' % (i+1))

    def img_color(i,c=True):
        pose_coords_per_frame = pose_coords[i]
        person_num = pose_coords_per_frame.shape[0]
        img_back = np.zeros((1080,1920,3),np.uint8)
        for j in range(person_num):
            img = add_joints( img_back,pose_coords_per_frame[j],c)
        
        return img

    img = img_color(i)*1 +img_color(j,False)*0.6+video[i]*0.6

    cv2.imwrite(img_path,img)
            




