import sys
sys.path.append("../common")

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.tsm.tsm_resnet import ResNetBackbone
from nets.module import Pose2Feat, Aggregator, Classifier
from nets.loss import CELoss, BCELoss
from config import cfg
import numpy as np


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
  
    # mask_012 =  np.round(dense_area_012, decimals =2) == np.round(quad_area_012[:,:,:,:,None,None], decimals=2)
    # mask_023 =  np.round(dense_area_023, decimals =2) == np.round(quad_area_023[:,:,:,:,None,None], decimals=2)
    mask_012 = dense_area_012 - quad_area_012[:,:,:,:,None,None]<=0.01
    mask_023 = dense_area_023 - quad_area_023[:,:,:,:,None,None]<=0.01

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

    refer_x = point2_x - point0_x
    refer_y = point2_y - point0_y
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


class Model(nn.Module):
    def __init__(self, img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num, skeleton):
        super(Model, self).__init__()
        self.img_backbone = img_backbone
        self.pose_backbone = pose_backbone
        self.pose2feat = pose2feat
        self.aggregator = aggregator
        self.classifier = classifier
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()

        self.class_num = class_num
        self.joint_num = joint_num
        self.skeleton_part = torch.LongTensor(skeleton).cuda().view(-1,2)
  
    def render_gaussian_heatmap(self, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1])
        y = torch.arange(cfg.input_hm_shape[0])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,None,:,:].cuda().float(); yy = yy[None,None,None,:,:].cuda().float();
        
        x = pose_coord[:,:,:,0,None,None]; y = pose_coord[:,:,:,1,None,None]; 
        heatmap = torch.exp(-(((xx-x)/cfg.hm_sigma)**2)/2 -(((yy-y)/cfg.hm_sigma)**2)/2) * (pose_score[:,:,:,None,None] > cfg.pose_score_thr).float() # score thresholding
        heatmap = heatmap.sum(1) # sum overall all persons
        heatmap[heatmap > 1] = 1 # threshold up to 1
        return heatmap
    
    def render_paf(self, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1])
        y = torch.arange(cfg.input_hm_shape[0])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,None,:,:].cuda().float(); yy = yy[None,None,None,:,:].cuda().float();
        
        # calculate vector between skeleton parts
        coord0 = pose_coord[:,:,self.skeleton_part[:,0],:] # batch_size*frame_num, person_num, part_num, 2
        coord1 = pose_coord[:,:,self.skeleton_part[:,1],:]
        vector = coord1 - coord0
        normalizer = torch.sqrt(torch.sum(vector**2,3,keepdim=True))
        normalizer[normalizer==0] = -1
        vector = vector / normalizer # normalize to unit vector
        vector_t = torch.stack((vector[:,:,:,1], -vector[:,:,:,0]),3)
        
        # make paf
        dist = vector[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None])
        dist_t = torch.abs(vector_t[:,:,:,0,None,None] * (xx - coord0[:,:,:,0,None,None]) + vector_t[:,:,:,1,None,None] * (yy - coord0[:,:,:,1,None,None]))
        mask1 = (dist >= 0).float(); mask2 = (dist <= normalizer[:,:,:,0,None,None]).float(); mask3 = (dist_t <= cfg.paf_sigma).float()
        score0 = pose_score[:,:,self.skeleton_part[:,0],None,None]
        score1 = pose_score[:,:,self.skeleton_part[:,1],None,None]
        mask4 = ((score0 >= cfg.pose_score_thr) * (score1 >= cfg.pose_score_thr)) # socre thresholding
        mask = mask1 * mask2 * mask3 * mask4 # batch_size*frame_num, person_num, part_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        paf = torch.stack((mask * vector[:,:,:,0,None,None], mask * vector[:,:,:,1,None,None]),3) # batch_size*frame_num, person_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        
        # sum and normalize
        mask = torch.sum(mask, (1))
        mask[mask==0] = -1
        paf = torch.sum(paf, (1)) / mask[:,:,None,:,:] # batch_size*frame_num, part_num, 2, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        paf = paf.view(paf.shape[0], paf.shape[1]*paf.shape[2], paf.shape[3], paf.shape[4])
        return paf 

    def render_limb_gaussian_heatmap(self, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1]) #[0,1,2, .., 55]
        y = torch.arange(cfg.input_hm_shape[0]) #[0,1,2, ..., 55]
        yy,xx = torch.meshgrid(y,x) #2: H,W
        xx = xx[None,None,None,:,:].cuda().float(); yy = yy[None,None,None,:,:].cuda().float() #5: 1, 1, 1, H,W

        # pose_coord = torch.LongTensor(pose_coord)

        # assert skeleton parts
        coord0 = pose_coord[:,:,self.skeleton_part[:,0],:] # 4: batch_size*frame_num, person_num, part_num, 2-(x,y)
        coord1 = pose_coord[:,:,self.skeleton_part[:,1],:] #  4: batch_size*frame_num, person_num, part_num, 2-(x,y)
        vector = coord1-coord0 #  4: batch_size*frame_num, person_num, part_num, 2-(x,y)

        dist_start = (xx - coord0[:,:,:,0,None,None])**2+ (yy - coord0[:,:,:,1,None,None])**2 #  5: batch_size*frame_num, person_num, part_num,H,W
        dist_end = (xx - coord1[:,:,:,0,None,None])**2+ (yy - coord1[:,:,:,1,None,None])**2  #  5: batch_size*frame_num, person_num, part_num,H,W
        dist_ab = (coord0[:,:,:,0,None,None]-coord1[:,:,:,0,None,None])**2+ (coord0[:,:,:,1,None,None]-coord1[:,:,:,1,None,None])**2
        dist_ab[dist_ab<1]=1 #  5: batch_size*frame_num, person_num, part_num,1,1

        coeff = (dist_start-dist_end+dist_ab)/2./dist_ab #  5: batch_size*frame_num, person_num, part_num,H,W

        a_dominate = coeff<=0
        b_dominate = coeff>=1
        seg_dominate =1- a_dominate.float() - b_dominate.float()  #5: batch_size*frame_num, person_num, part_num,H,W

        projection_x =coeff*vector[:,:,:,0,None,None]+coord0[:,:,:,0,None,None]
        projection_y =coeff*vector[:,:,:,1,None,None]+coord0[:,:,:,1,None,None]
        d2_line = (xx-projection_x)**2+(yy-projection_y)**2 #5: batch_size*frame_num, person_num, part_num,H,W
        d2_seg = a_dominate*dist_start +b_dominate*dist_end+ seg_dominate*d2_line

        heatmap = torch.exp(-d2_seg/2./cfg.hm_sigma**2)* (pose_score[:,:,self.skeleton_part[:,0],None,None] > cfg.pose_score_thr)
        heatmap = heatmap.sum(1)
        heatmap[heatmap>1]=1 #4: batch_size*frame_num, part_num,H,W
        return heatmap

    def render_limb_motion(self, batch_size, pose_coord, pose_score):
        x = torch.arange(cfg.input_hm_shape[1]) #[0,1,2, .., 55]
        y = torch.arange(cfg.input_hm_shape[0]) #[0,1,2, ..., 55]
        yy,xx = torch.meshgrid(y,x) #2: H,W
        xx = xx[None,None,None,None,:,:].cuda().float(); yy = yy[None,None,None,None,:,:].cuda().float() #6: 1, 1, 1, 1, H,W
        # pose_coord = torch.LongTensor(pose_coord)
        # reshape pose_coord like 5: batch_size, frame_num, person_num, part_num, 2
        frame_num = int(pose_coord.shape[0]/batch_size) 
        motion_sequence =  torch.LongTensor([[i, i+1] for i in range(frame_num-1)]).cuda().view(-1,2)
        pose_coord = pose_coord.view( batch_size,frame_num, pose_coord.shape[1], pose_coord.shape[2], pose_coord.shape[3])
        #get four point for temp0/1_coord0/1
        temp0 = pose_coord[:, motion_sequence[:,0], :, :, :] # 5: batch_size, adj_num, person_num, part_num, 2
        temp1 = pose_coord[:, motion_sequence[:,1], :, :, :] # 5: batch_size, adj_num, person_num, part_num, 2
    
        temp0_coord0 = temp0[:, :, :, self.skeleton_part[:,0], :] # 5: batch_size, adj_num, person_num, part_num, 2
        temp0_coord1 = temp0[:, :, :, self.skeleton_part[:,1], :] # 5: batch_size, adj_num, person_num, part_num, 2
        temp1_coord0 = temp1[:, :, :, self.skeleton_part[:,0], :] # 5: batch_size, adj_num, person_num, part_num, 2
        temp1_coord1 = temp1[:, :, :, self.skeleton_part[:,1], :] # 5: batch_size, adj_num, person_num, part_num, 2
        #get pose score for each joint, pose_score 3: batch_size*frame_num, person_num, joint_num
        #pose_score = torch.LongTensor(pose_score)
        pose_score = pose_score.view(batch_size, frame_num, pose_score.shape[1], pose_score.shape[2])
        pose_score_0 = pose_score[:,motion_sequence[:,0],:,:]
        pose_score_0_0 = pose_score_0[:,:,:,self.skeleton_part[:,0], None, None]
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
        co_cross_mask = line_cross_mask.float()*temp_cross_mask.float()
        co_cross_mask = 1.0 - co_cross_mask
        none_cross_mask = (line_cross_mask+temp_cross_mask)>0.0
        none_cross_mask = 1.0 - none_cross_mask.float()

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
                                            + co_cross_mask[:,:,:,:,None,None]*line_cross_mask[:,:,:,:,None,None]*(line_coord0_mask*velocity_coord0[:,:,:,:,0,None,None]+ line_coord1_mask*velocity_coord1[:,:,:,:,0,None,None])
        limb_motion_y = none_cross_mask[:,:,:,:,None,None]*none_quad_mask*velocity_mean[:,:,:,:,1,None,None] \
                                            + temp_cross_mask[:,:,:,:,None,None]*(temp_quad_mask*velocity_coord0[:,:,:,:,1,None,None] + temp_circle_mask*velocity_coord1[:,:,:,:,1,None,None]) \
                                            + co_cross_mask[:,:,:,:,None,None]*line_cross_mask[:,:,:,:,None,None]*(line_coord0_mask*velocity_coord0[:,:,:,:,1,None,None] + line_coord1_mask*velocity_coord1[:,:,:,:,1,None,None])
        
        #fuse pose_score batch_size, adj_num, person_num, part_num,H,W
        limb_motion_x = mask_pose_score*limb_motion_x
        limb_motion_y = mask_pose_score*limb_motion_y
        limb_motion = torch.stack((limb_motion_x,limb_motion_y),4)
        # batch_size, adj_num, person_num, part_num,2,H,W
        limb_motion = limb_motion.sum(2)
        # batch_size, adj_num,  part_num,2,H,W
        limb_motion = torch.cat((limb_motion,limb_motion.mean(1)[:,None,:,:,:,:]), 1)
        # batch_size, frame_num, part_num, 2, H, W
        normlizer = torch.sqrt(limb_motion[:,:,:,0,:,:]**2+limb_motion[:,:,:,1,:,:]**2)
        normlizer[normlizer==0] = 1
        limb_motion = limb_motion/normlizer[:,:,:,None,:,:]
        # batch_size, frame_num, part_num, 2, H, W

        limb_motion = limb_motion.view(limb_motion.shape[0]*limb_motion.shape[1],limb_motion.shape[2]*limb_motion.shape[3],limb_motion.shape[4],limb_motion.shape[5])
        # limb_motion 4: batch_size*frame_num, 2*part_num, H, W
        return limb_motion


    def forward(self, inputs, targets, meta_info, mode):
        input_video = inputs['video'] # batch_size, frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]
        batch_size, video_frame_num = input_video.shape[:2]
        input_video = input_video.view(batch_size*video_frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1])

        pose_coords = inputs['pose_coords'] # batch_size, frame_num, person_num, joint_num, 2
        pose_scores = inputs['pose_scores'] # batch_size, frame_num, person_num, joint_num
        batch_size, pose_frame_num = pose_coords.shape[:2]
        pose_coords = pose_coords.view(batch_size*pose_frame_num, cfg.top_k_pose, self.joint_num, 2)
        pose_scores = pose_scores.view(batch_size*pose_frame_num, cfg.top_k_pose, self.joint_num)
        input_pose_hm = self.render_gaussian_heatmap(pose_coords, pose_scores) # batch_size*pose_frame_num, self.joint_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]
        #input_pose_paf = self.render_paf(pose_coords, pose_scores) # batch_size*pose_frame_num, 2*part_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]

        input_pose_limb_hm = self.render_limb_gaussian_heatmap(pose_coords, pose_scores) # batch_size*frame_num, part_num, H, W
        input_pose_limb_mo = self.render_limb_motion(batch_size, pose_coords, pose_scores) # batch_size*frame_num, 2*part_num, H, W

        # rgb only
        if cfg.mode == 'rgb_only':
            video_feat = self.img_backbone(input_video, skip_early=False)
            action_label_out = self.classifier(video_feat)
            action_label_out = action_label_out.view(batch_size, video_frame_num, -1).mean(1)
        # pose only
        elif cfg.mode == 'pose_only':
            pose_feat = self.pose2feat(input_pose_hm, input_pose_limb_hm, input_pose_limb_mo)
            pose_feat = self.pose_backbone(pose_feat, skip_early=True)
            action_label_out = self.classifier(pose_feat)
            action_label_out = action_label_out.view(batch_size, pose_frame_num, -1).mean(1)
        # pose late fusion
        elif cfg.mode == 'rgb+pose':
            video_feat = self.img_backbone(input_video, skip_early=False)
            pose_feat = self.pose2feat(input_pose_hm, input_pose_limb_hm, input_pose_limb_mo)
            pose_feat = self.pose_backbone(pose_feat, skip_early=True)
            video_pose_feat, pose_gate, video_gate = self.aggregator(video_feat, pose_feat)
            action_label_out = self.classifier(video_pose_feat)
            action_label_out = action_label_out.view(batch_size, video_frame_num, -1).mean(1)
            
        if mode == 'train':
            # loss functions
            loss = {}
            loss['action_cls'] = self.ce_loss(action_label_out, targets['action_label'])
            if cfg.mode == 'rgb+pose' and cfg.pose_gate: 
                loss['pose_gate'] = self.bce_loss(pose_gate, torch.zeros_like(pose_gate)+0.5) * cfg.reg_weight
            return loss

        else:
            # test output
            out = {}
            out['action_prob'] = F.softmax(action_label_out,1)
            out['img_id'] = meta_info['img_id']
            if cfg.mode == 'rgb+pose' and cfg.pose_gate: out['pose_gate'] = pose_gate.view(batch_size, -1).mean((1))
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(class_num, joint_num, skeleton, mode):
    img_backbone = ResNetBackbone(cfg.img_resnet_type, cfg.frame_per_seg)
    pose_backbone = ResNetBackbone(cfg.pose_resnet_type, (cfg.frame_per_seg-1)*cfg.pose_frame_factor+1)
    pose2feat = Pose2Feat(joint_num, skeleton)
    aggregator = Aggregator()
    classifier = Classifier(class_num)

    if mode == 'train':
        img_backbone.init_weights()
        pose_backbone.init_weights()
        pose2feat.apply(init_weights)
        aggregator.apply(init_weights)
        classifier.apply(init_weights)
   
    model = Model(img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num, skeleton)
    return model

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

if __name__ == "__main__":
    img_backbone = ResNetBackbone(18, 8)
    pose_backbone = ResNetBackbone(18, 8)
    aggregator = Aggregator()

    skeleton = skeleton = ( (1,0), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),(1,8), (8,9), (9,10),(1,11), (11,12), (12,13))
    pose2feat = Pose2Feat(18, skeleton)
    print_model_parm_nums(img_backbone)
    print_model_parm_nums(pose2feat)
    print_model_parm_nums(pose_backbone)
    print_model_parm_nums(aggregator)
