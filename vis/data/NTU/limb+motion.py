
from curses import noecho
from tkinter.messagebox import NO
from turtle import shape
from PIL import Image
import os
import cv2
from cv2 import waitKey
import numpy as np
import math 
import random
import os.path as osp
import torch
import torchaudio

class Cfg():
    input_img_shape =  (1080, 1920)
    input_hm_shape =  (112,112)
    hm_sigma = 10
    paf_sigma = 10
    pose_score_thr = 0.6
    batch_size =1
    joint_num = 25
    frame_num = 8
    person_num =5
    skeleton =((0,20), (20,3),(20,4),(4,5),(5,6),(20,8),(8,9),(9,10),(12,13),(13,14),(16,17),(17,18))
    skeleton_part=np.array(skeleton)
    part_num = skeleton_part.shape[0]

    motion_sequence =np.array([[i, i+1] for i in range(frame_num-1)]) 


def load_img(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = img.resize(cfg.input_img_shape[0],cfg.input_img_shape[1])

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

def get_line_intersection(point0,point1, point2, point3):

    #input
    #point 5:bath_size, adj_num, person_num, part_num,2
    #line segment point0 -> point1, point2->point3

    #output
    #mask 4:batch_size, adj_num, person_num, part_num
    #cross point 5: batch_size, adj_num, person_num, part_num, 2

    #get basic 4:batch_size, adj_num, person_num, part_num
    s10_x = point1[:,:,:,:,0] - point0[:,:,:,:,0]
    s10_y = point1[:,:,:,:,1] - point0[:,:,:,:,1]
    s32_x = point3[:,:,:,:,0] - point2[:,:,:,:,0]
    s32_y = point3[:,:,:,:,1] - point2[:,:,:,:,1]

    mask = torch.ones_like(s10_x)

    #case 1
    denom = s10_x*s32_y - s32_x*s10_y
    denomPositive = (denom>0).float()
    denomNegative = (denom<0).float()

    mask[denom==0]=0.0

    #case 2
    s02_x = point0[:,:,:,:,0] - point2[:,:,:,:,0]
    s02_y = point0[:,:,:,:,1] - point2[:,:,:,:,1]

    s_numer = s10_x*s02_y - s10_y*s02_x
    s_numer_Positive =(s_numer<0).float()
    s_numer_Negative = (s_numer>0).float()
    mask = mask*(1- s_numer_Positive*denomPositive)*(1- s_numer_Negative*denomNegative)

    #case 3
    t_numer = s32_x * s02_y - s32_y * s02_x
    t_numer_Positive =(t_numer<0).float()
    t_numer_Negative = (t_numer>0).float()
    mask = mask*(1-t_numer_Positive*denomPositive)*(1- t_numer_Negative*denomNegative)

    #case 4
    sd_Positive = (s_numer>denom).float()
    sd_Negative = (s_numer<denom).float()
    mask = mask*(1-sd_Positive*denomPositive)*(1-sd_Negative*denomNegative)
    #case 5
    td_Positive = (t_numer>denom).float()
    td_Negative = (t_numer<denom).float()
    mask = mask*(1-td_Positive*denomPositive)*(1-td_Negative*denomNegative)

    denom[denom==0]=-1
    t = t_numer/denom
    cross_point_x = point0[:,:,:,:,0] + t*s10_x
    cross_point_y = point0[:,:,:,:,1] + t*s10_y

    cross_point = torch.stack((cross_point_x, cross_point_y),4)

    return mask, cross_point

def get_triangle_dense_area(xx, yy, point0, point1):
    #input point: 
    #s : batch_size, adj_num, person_num, part_num,H,W
    point0_x = point0[:,:,:,:,0,None,None]
    point0_y = point0[:,:,:,:,1,None,None]
    point1_x = point1[:,:,:,:,0,None,None]
    point1_y = point1[:,:,:,:,1,None,None]

    edge_a = torch.sqrt((point0_x- point1_x)**2+(point0_y-point1_y)**2)
    edge_b = torch.sqrt((xx-point0_x)**2+(yy-point0_y)**2)
    edge_c = torch.sqrt((xx-point1_x)**2+(yy-point1_y)**2)
    mean_p = (edge_a+edge_b+edge_c)/2
    area_ = torch.sqrt(mean_p*(mean_p-edge_a)*(mean_p-edge_b)*(mean_p-edge_c))
    return area_

def get_triangle_area(point0, point1, point2):
    #input point: 
    #s: batch_size, adj_num, person_num, part_num
    point0_x = point0[:,:,:,:,0]
    point0_y = point0[:,:,:,:,1]
    point1_x = point1[:,:,:,:,0]
    point1_y = point1[:,:,:,:,1]
    point2_x = point2[:,:,:,:,0]
    point2_y = point2[:,:,:,:,1]

    edge_a= torch.sqrt((point0_x-point1_x)**2+(point0_y-point1_y)**2)
    edge_b= torch.sqrt((point0_x-point2_x)**2+(point0_y-point2_y)**2)
    edge_c= torch.sqrt((point1_x-point2_x)**2+(point1_y-point2_y)**2)
    mean_p = (edge_a+edge_b+edge_c)/2
    area_ = torch.sqrt(mean_p*(mean_p-edge_a)*(mean_p-edge_b)*(mean_p-edge_c))
    return area_

def get_quad_region(xx, yy, point0, point1, point2, point3):
    #input
    #xx 6: 1, 1, 1, 1, H, W
    #point 5:batch_size, adj_num, person_num, part_num,2
    #coord0[:,:,:, :,0,None,None]
    #coord 6: batch_size, adj_num, person_num, part_num, None, None
    #quadrilateral  point0 -> point1-> point2->point3
    quad_area_012 = get_triangle_area(point0,point1,point2)
    quad_area_023 = get_triangle_area(point0,point2,point3)
    dense_area_012 = get_triangle_dense_area(xx,yy,point0,point1)+get_triangle_dense_area(xx,yy,point1,point2)+get_triangle_dense_area(xx,yy,point2,point0)
    dense_area_023 = get_triangle_dense_area(xx,yy,point2,point3)+get_triangle_dense_area(xx,yy,point3,point0)+get_triangle_dense_area(xx,yy,point2,point0)

    #exclude small area such like 0
    mask_area_012 = quad_area_012>0
    mask_area_023 = quad_area_023>0
  
    mask_012 =  np.round(dense_area_012, decimals =2) == np.round(quad_area_012[:,:,:,:,None,None], decimals=2)
    mask_023 =  np.round(dense_area_023, decimals =2) == np.round(quad_area_023[:,:,:,:,None,None], decimals=2)

    mask = (mask_area_012[:,:,:,:,None,None].float())*mask_012.float()+(mask_area_023[:,:,:,:,None,None].float())*mask_023.float()
    mask[mask>=1.0]=1.0
    return mask

def get_cos_angles(vector_x,vector_y, refer_x, refer_y):
    vector_m = torch.sqrt(vector_x**2+vector_y**2)
    refer_m = torch.sqrt(refer_x**2+refer_y**2)
    m_mix = vector_m*refer_m
    m_mix [m_mix==0]=1
    val = (vector_x*refer_x+vector_y*refer_y)/m_mix
    val[val>1]=1
    val[val<-1]=-1
    angle = torch.acos(val)
    return angle

def get_circle_region(xx, yy, point0, point1, point2,):
    #input
    #xx 6: 1, 1, 1, 1, H, W
    #point 5:batch_size, adj_num, person_num, part_num,2
    point0_x = point0[:,:,:,:,0,None,None]
    point0_y = point0[:,:,:,:,1,None,None]
    point1_x = point1[:,:,:,:,0,None,None]
    point1_y = point1[:,:,:,:,1,None,None]
    point2_x = point2[:,:,:,:,0,None,None]
    point2_y = point2[:,:,:,:,1,None,None]

    refer_x = point1_x - point0_x
    refer_y = point1_y - point0_y
    end_x = point2_x - point0_x
    end_y = point2_y - point0_y
    dense_x = xx - point0_x
    dense_y = yy - point0_y

    refer_r = torch.sqrt((refer_x)**2+(refer_y)**2)
    end_r = torch.sqrt((end_x)**2+(end_y)**2)
    thre_r = torch.max(refer_r,end_r)
    thre_angles = get_cos_angles(end_x,end_y, refer_x,refer_y)

    dense_raduis = torch.sqrt((dense_x)**2+(dense_y)**2)
    dense_angles = get_cos_angles(dense_x,dense_y,refer_x,refer_y)

    mask_1 = dense_raduis<=thre_r
    mask_2 = dense_angles<=thre_angles
    mask = mask_1.float()*mask_2.float()
    return mask


def render_limb_motion(cfg, pose_coord, pose_score):
    x = torch.arange(cfg.input_hm_shape[1]) #[0,1,2, .., 55]
    y = torch.arange(cfg.input_hm_shape[0]) #[0,1,2, ..., 55]
    yy,xx = torch.meshgrid(y,x) #2: H,W
    xx = xx[None,None,None,None,:,:]; yy = yy[None,None,None,None,:,:] #6: 1, 1, 1, 1, H,W
    pose_coord = torch.LongTensor(pose_coord)
    # reshape pose_coord like 5: batch_size, frame_num, person_num, part_num, 2
    pose_coord = pose_coord.view( cfg.batch_size, cfg.frame_num, pose_coord.shape[1], pose_coord.shape[2], pose_coord.shape[3])
    #get four point for temp0/1_coord0/1
    temp0 = pose_coord[:, cfg.motion_sequence[:,0], :, :, :] # 5: batch_size, adj_num, person_num, part_num, 2
    temp1 = pose_coord[:, cfg.motion_sequence[:,1], :, :, :] # 5: batch_size, adj_num, person_num, part_num, 2
   
    temp0_coord0 = temp0[:, :, :, cfg.skeleton_part[:,0], :] # 5: batch_size, adj_num, person_num, part_num, 2
    temp0_coord1 = temp0[:, :, :, cfg.skeleton_part[:,1], :] # 5: batch_size, adj_num, person_num, part_num, 2
    temp1_coord0 = temp1[:, :, :, cfg.skeleton_part[:,0], :] # 5: batch_size, adj_num, person_num, part_num, 2
    temp1_coord1 = temp1[:, :, :, cfg.skeleton_part[:,1], :] # 5: batch_size, adj_num, person_num, part_num, 2
    #get pose score for each joint, pose_score 3: batch_size*frame_num, person_num, joint_num
    pose_score = torch.LongTensor(pose_score)
    pose_score = pose_score.view(cfg.batch_size, cfg.frame_num, pose_score.shape[1], pose_score.shape[2])
    pose_score_0 = pose_score[:,cfg.motion_sequence[:,0],:,:]
    pose_score_0_0 = pose_score_0[:,:,:,cfg.skeleton_part[:,0], None, None]
    # pose_score_0_0:batch_size, adj_num, person_num, part_num,1,1
    mask_pose_score = pose_score_0_0>=cfg.pose_score_thr

    #get two velocity vector for coord0 (means start) and coord1 (means end)

    velocity_coord0 = (temp1_coord0 - temp0_coord0)
    velocity_coord1 = (temp1_coord1 - temp0_coord1)
    velocity_mean = (velocity_coord0+velocity_coord1)
    
    #divide each limb part into 3 cases with part mask
    #batch_size, adj_num, person_num, part_num, 1
    line_cross_mask, line_cross_point = get_line_intersection(temp0_coord0, temp0_coord1, temp1_coord0, temp1_coord1)
    temp_cross_mask, temp_cross_point = get_line_intersection(temp0_coord0, temp1_coord0, temp0_coord1, temp1_coord1)
    none_cross_mask = 1 - line_cross_mask - temp_cross_mask

    #region mask
    #batch_size, adj_num, person_num, part_num,H,W
    #case 1, velocity_none_quad = velocity_mean
    none_quad_mask = get_quad_region(xx, yy, temp0_coord0, temp0_coord1, temp1_coord1, temp1_coord0)

    #case 2, velocity_temp_quad = velocity_coord0, velocity_temp_sector = velocity_coord1
    temp1_coord1_extend = temp0_coord1+velocity_coord0
    temp_quad_mask = get_quad_region(xx, yy, temp0_coord0, temp0_coord1, temp1_coord1_extend,temp1_coord0)
    temp_circle_mask = get_circle_region(xx, yy, temp1_coord0, temp1_coord1_extend, temp1_coord1)

    #case 3, velocity_line_coord0 = velocity_coord0, velocity_line_coord1 = velocity_coord1
    line_coord0_mask = get_circle_region(xx, yy, line_cross_point, temp0_coord0, temp1_coord0)
    line_coord1_mask = get_circle_region(xx, yy, line_cross_point, temp0_coord1, temp1_coord1)

    #fusion all case
    limb_motion_x = none_cross_mask[:,:,:,:,None,None]*none_quad_mask*velocity_mean[:,:,:,:,0,None,None] \
                                        + temp_cross_mask[:,:,:,:,None,None]*(temp_quad_mask*velocity_coord0[:,:,:,:,0,None,None] + temp_circle_mask*velocity_coord1[:,:,:,:,0,None,None]) \
                                        + line_cross_mask[:,:,:,:,None,None]*(line_coord0_mask*velocity_coord0[:,:,:,:,0,None,None]+ line_coord1_mask*  velocity_coord1[:,:,:,:,0,None,None])
    limb_motion_y =  none_cross_mask[:,:,:,:,None,None]*none_quad_mask*velocity_mean[:,:,:,:,1,None,None] \
                                        + temp_cross_mask[:,:,:,:,None,None]*(temp_quad_mask*velocity_coord0[:,:,:,:,1,None,None] + temp_circle_mask*velocity_coord1[:,:,:,:,1,None,None]) \
                                        + line_cross_mask[:,:,:,:,None,None]*(line_coord0_mask*velocity_coord0[:,:,:,:,1,None,None] + line_coord1_mask*  velocity_coord1[:,:,:,:,1,None,None])
    
    #fuse pose_score batch_size, adj_num, person_num, part_num,H,W
   #fuse pose_score batch_size, adj_num, person_num, part_num,H,W
    limb_motion_x = mask_pose_score*limb_motion_x
    limb_motion_y = mask_pose_score*limb_motion_y
    limb_motion = torch.stack((limb_motion_x,limb_motion_y),4)
    # batch_size, adj_num, person_num, part_num,2,H,W
    limb_motion = limb_motion.sum(2)
    # batch_size, adj_num,  part_num,2,H,W
    limb_motion = torch.cat((limb_motion,limb_motion.sum(1)[:,None,:,:,:,:]), 1)
    # batch_size, frame_num, part_num, 2, H, W
    normlizer = torch.sqrt(limb_motion[:,:,:,0,:,:]**2+limb_motion[:,:,:,1,:,:]**2)
    normlizer[normlizer==0] =1
    limb_motion = limb_motion/normlizer[:,:,:,None,:,:]
    # batch_size, frame_num, part_num, 2, H, W

    limb_motion = limb_motion.view(limb_motion.shape[0], limb_motion.shape[1],limb_motion.shape[2]*limb_motion.shape[3],limb_motion.shape[4],limb_motion.shape[5])
    # limb_motion 4: batch_size*frame_num, 2*part_num, H, W
    return limb_motion


if __name__ =="__main__":

    cfg =Cfg()


    skeleton_path =  "/home/gft/PycharmProjects/vis/data/NTU/samples/S013C001P028R002A058.skeleton"
    img_path = "/home/gft/PycharmProjects/vis/data/NTU/samples/S013C001P028R002A058_rgb"
    save_path = "/home/gft/PycharmProjects/vis/data/NTU/figs/S013C001P028R002A058/limb_motion_heatmap"

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    frame_num = [1,8, 15, 21, 29, 37, 45, 50]
    top_k_pose=5
    joint_num = 25
    part_num =24
    joint_names = ('Pelvis', 'Chest', 'Neck', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand1', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand1', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot', 'Thorax', 'R_Hand2', 'R_Hand3', 'L_Hand2', 'L_Hand3')
    flip_pairs = ( (4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18), (15,19), (21,23), (22,24) )
    skeleton = ( (0,1), (1,2), (2,3), (4,20), (4,5), (5,6), (6,7), (7,21), (21,22), (8,20), (8,9), (9,10), (10,11), (11,23), (23,24), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19), (1,20) )


    video, frame_idxs =load_video(img_path,frame_num)
    
    pose_coords, pose_scores = load_skeleton(skeleton_path,frame_idxs,cfg.input_img_shape , cfg.input_img_shape )

    pose_coords, pose_scores = process_skeleton(cfg, pose_coords, pose_scores,joint_num)

    limb_motion_heatmap = render_limb_motion(cfg, pose_coords, pose_scores)

    print(limb_motion_heatmap.shape)


    def limb_motion_color(pose_paf):

        pose_paf = pose_paf.numpy()
        print(np.min(pose_paf),np.max(pose_paf))
        for j in range(8):
            pose_paf_ = np.zeros_like(pose_paf[0][0][0])
            for i in range(24):
                pose_paf_i =(pose_paf[0][j][i]-1)/2*255
                pose_paf_ = pose_paf_+pose_paf_i

            pose_paf_[pose_paf_>255]=255
            pose_paf_=cv2.applyColorMap(pose_paf_.astype(np.uint8),11)
            cv2.imwrite(osp.join(save_path,'pose_paf_%s.jpg'%(j)),pose_paf_)

    limb_motion_color(limb_motion_heatmap)


    
    



  
   
        




