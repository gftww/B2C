
from PIL import Image
import os
import cv2
from cv2 import waitKey
import numpy as np
import math 
import random
import os.path as osp
import torch

class Cfg():
    input_img_shape = (1080,1920)
    input_hm_shape = (1080,1920)
    hm_sigma = 15
    paf_sigma = 10
    pose_score_thr = 0.1
    skeleton =( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )
    skeleton_part=np.array(skeleton)



def load_img(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    return img

def load_video(path, frame_num):
    
    # load frames
    video = []
    pose_frame_idxs=[]
    for i in frame_num:
    
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

    return:number of people P=5,J =25
    pose_coords = T*P*J*2
    pose_scores = T*P*J
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

def add_joints(image, joints):
    coco_part_labels = ['Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3']

    coco_part_orders = [ (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20)]

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

def process_skeleton(cfg, pose_coords, pose_scores,  joint_num):
    frame_num, person_num = pose_coords.shape[:2]
    pose_coords = pose_coords.reshape(-1,2)
    pose_scores = pose_scores.reshape(-1)


    # transform to input heatmap space
    # transform to input heatmap space
    pose_coords[:,0] = pose_coords[:,0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
    pose_coords[:,1] = pose_coords[:,1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]

    pose_coords = pose_coords.reshape(frame_num, person_num, joint_num, 2)
    pose_scores = pose_scores.reshape(frame_num, person_num, joint_num)
    
    return pose_coords, pose_scores


def render_gaussian_heatmap(cfg,pose_coord, pose_score):
    x = torch.arange(cfg.input_hm_shape[1])
    y = torch.arange(cfg.input_hm_shape[0])
    yy,xx = torch.meshgrid(y,x)
    xx = xx[None,None,None,:,:]; yy = yy[None,None,None,:,:]
    
    x = pose_coord[:,:,:,0,None,None]; y = pose_coord[:,:,:,1,None,None]; 
    heatmap = torch.exp(-(((xx-x)/cfg.hm_sigma)**2)/2 -(((yy-y)/cfg.hm_sigma)**2)/2) * (pose_score[:,:,:,None,None] > cfg.pose_score_thr)# score thresholding
    heatmap = heatmap.sum(1) # sum overall all persons
    heatmap[heatmap > 1] = 1 # threshold up to 1
    return heatmap

def render_paf(cfg, pose_coord, pose_score):
    x = torch.arange(cfg.input_hm_shape[1])
    y = torch.arange(cfg.input_hm_shape[0])
    yy,xx = torch.meshgrid(y,x)
    xx = xx[None,None,None,:,:]; yy = yy[None,None,None,:,:]
    
    # calculate vector between skeleton parts
    coord0 = pose_coord[:,:,cfg.skeleton_part[:,0],:] # batch_size*frame_num, person_num, part_num, 2
    coord1 = pose_coord[:,:,cfg.skeleton_part[:,1],:]
    vector = coord1 - coord0
    vector = torch.LongTensor(vector)
    normalizer = torch.sqrt(torch.sum(vector**2,3,keepdim=True))
    normalizer[normalizer==0] = -1
    vector = vector / normalizer # normalize to unit vector
    vector_t = torch.stack((vector[:,:,:,1], -vector[:,:,:,0]),3)
    
    # make paf
    dist = vector[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None])
    dist_t = torch.abs(vector_t[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector_t[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None]))
    mask1 = (dist >= 0).float(); mask2 = (dist <= normalizer[:,:,:,0,None,None]).float(); mask3 = (dist_t <= cfg.paf_sigma).float()
    score0 = pose_score[:,:,cfg.skeleton_part[:,0],None,None]
    score1 = pose_score[:,:,cfg.skeleton_part[:,1],None,None]
    mask4 = ((score0 >= cfg.pose_score_thr) * (score1 >= cfg.pose_score_thr)) # socre thresholding
    mask = mask1 * mask2 * mask3 * mask4 # batch_size*frame_num, person_num, part_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
    paf = torch.stack((mask * vector[:,:,:,0,None,None], mask * vector[:,:,:,1,None,None]),3) # batch_size*frame_num, person_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
    
    # sum and normalize
    mask = torch.sum(mask, (1))
    mask[mask==0] = -1
    paf = torch.sum(paf, (1)) / mask[:,:,None,:,:] # batch_size*frame_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
    paf = paf.view(paf.shape[0], paf.shape[1]*paf.shape[2], paf.shape[3], paf.shape[4]) # batch_size*frame_num, part_num*2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
    return paf


if __name__ =="__main__":

    cfg =Cfg()

    skeleton_path =  "/home/zj-user/Icode/vis/data/NTU/samples/S013C001P027R002A028.skeleton"
    img_path = "/home/zj-user/Icode/vis/data/NTU/samples/S013C001P027R002A028_rgb"
    save_path = "/home/zj-user/Icode/vis/data/NTU/figs/S013C001P027R002A028/pose_heatmap_54"



    if not os.path.exists(save_path):
        os.makedirs(save_path)


    frame_num = [54]
    top_k_pose=5

    class_num = 60
    joint_num = 25
    joint_names = ('Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3')
    flip_pairs = ( (4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18), (15,19), (21,23), (22,24) )
    skeleton = ( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )


    video, frame_idxs =load_video(img_path,frame_num)
    
    pose_coords, pose_scores = load_skeleton(skeleton_path,frame_idxs,cfg.input_img_shape , cfg.input_img_shape )

    pose_coords, pose_scores = process_skeleton(cfg, pose_coords, pose_scores,joint_num)

    heatmap =render_gaussian_heatmap(cfg,pose_coords, pose_scores)
    pose_paf = render_paf(cfg, pose_coords, pose_scores) 

    def heatmap_color(heatmap):
        heatmap = heatmap.numpy()
        heatmap_ = np.zeros_like(heatmap[ 0][0])
        for i in range(joint_num):
            heatmap_i =heatmap[0][i][:]*255
            heatmap_ = heatmap_+heatmap_i

            heatmap_i = cv2.applyColorMap(heatmap_i.astype(np.uint8),2)

            cv2.imwrite(osp.join(save_path,'heatmap_%s.jpg'%(i)),heatmap_i)

        heatmap_[heatmap_>255]=255
        heatmap_=cv2.applyColorMap(heatmap_.astype(np.uint8),2)

        cv2.imwrite(osp.join(save_path,'heatmap_%s.jpg'%(25)),heatmap_)

    def heatmap_img(heatmap, img):
        heatmap = heatmap.numpy()
        heatmap_ = np.zeros_like(heatmap[ 0][0])

        for i in range(joint_num):
            heatmap_i =heatmap[0][i][:]*255
            heatmap_ = heatmap_+heatmap_i

            heatmap_i = cv2.applyColorMap(heatmap_i.astype(np.uint8),2)
            heatmap_i = heatmap_i*0.4+img*0.6
            cv2.imwrite(osp.join(save_path,'heatmap_img_%s.jpg'%(i)),heatmap_i)

        heatmap_[heatmap_>255]=255
        heatmap_=cv2.applyColorMap(heatmap_.astype(np.uint8),2)
        heatmap_ = heatmap_*0.4+img*0.6
        cv2.imwrite(osp.join(save_path,'heatmap_img_%s.jpg'%(25)),heatmap_)



    def pose_paf_color(pose_paf):

        pose_paf = pose_paf.numpy()
        pose_paf_ = np.zeros_like(pose_paf[0][0])

        print(np.min(pose_paf), np.max(pose_paf))
        for i in range(48):
            pose_paf_i =pose_paf[0][i][:]*255
            pose_paf_ = pose_paf_+pose_paf_i
    
            pose_paf_i = cv2.applyColorMap(pose_paf_i.astype(np.uint8),2)
            cv2.imwrite(osp.join(save_path,'pose_paf_%s.jpg'%(i)),pose_paf_i)

        pose_paf_[pose_paf_>255]=255

        pose_paf_=cv2.applyColorMap(pose_paf_.astype(np.uint8),2)
        cv2.imwrite(osp.join(save_path,'pose_paf_%s.jpg'%(49)),pose_paf_)

    def pose_paf_img(pose_paf,  img):

        pose_paf = pose_paf.numpy()
        pose_paf_ = np.zeros_like(pose_paf[0][0])
        for i in range(48):
            pose_paf_i =pose_paf[0][i][:]*255
            pose_paf_ = pose_paf_+pose_paf_i
    
            pose_paf_i = cv2.applyColorMap(pose_paf_i.astype(np.uint8),2)
            pose_paf_i = pose_paf_i*0.4+img*0.6
            cv2.imwrite(osp.join(save_path,'pose_paf_img_%s.jpg'%(i)),pose_paf_i)

        pose_paf_[pose_paf_>255]=255
        pose_paf_=cv2.applyColorMap(pose_paf_.astype(np.uint8),2)
        pose_paf_ = pose_paf_*0.4+img*0.6
        cv2.imwrite(osp.join(save_path,'pose_paf_img_%s.jpg'%(49)),pose_paf_)

    heatmap_color(heatmap)
    heatmap_img(heatmap,video[0])
    pose_paf_color(pose_paf)
    pose_paf_img(pose_paf,video[0])
    
    



  
   
        




