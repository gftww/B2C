import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_conv_layers, make_conv3d_layers, make_linear_layers, make_convtrans3d_layers, make_convtrans_layers

class Pose2Feat(nn.Module):
    def __init__(self, joint_num, skeleton):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.skeleton_num = len(skeleton)
        #self.input_dim = self.joint_num+3*self.skeleton_num
        #self.input_dim = self.joint_num
        #self.input_dim = self.skeleton_num
        #self.input_dim = self.joint_num+2*self.skeleton_num
        #self.input_dim = 3*self.skeleton_num
        self.input_dim = self.joint_num+4*self.skeleton_num

        self.data_bn = nn.BatchNorm2d(self.input_dim)
        self.conv = make_conv_layers([self.input_dim,64])

    def forward(self, input_pose_hm, input_pose_limb_hm, input_pose_limb_mo):
        pose_feat = torch.cat((input_pose_hm, input_pose_limb_hm, input_pose_limb_mo),1)
        pose_feat = self.data_bn(pose_feat)
        pose_feat = self.conv(pose_feat)
        return pose_feat


class Aggregator_0(nn.Module):
    def __init__(self):
        super(Aggregator_0, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        ## pose gate layer
        self.pose_gate_fc = make_linear_layers([self.pose_resnet_dim, cfg.agg_feat_dim], relu_final=False)

        ## aggregation layer
        self.img_conv = make_conv_layers([self.img_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
        self.img_norm = nn.LayerNorm([cfg.agg_feat_dim, 1, 1])
        self.pose_conv = make_conv_layers([self.pose_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
        self.pose_norm = nn.LayerNorm([cfg.agg_feat_dim, 1, 1])


    def forward(self, video_feat, pose_feat):
        pose_feat = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]

        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
            batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0] // self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
            pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
            pose_feat = self.pose_temporal_conv(pose_feat)
            pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)
       
        # pose gate estimator
        if cfg.pose_gate:
            pose_gate = torch.sigmoid(self.pose_gate_fc(torch.squeeze(pose_feat)))[:,:,None,None]
            
            pose_feat = self.pose_conv(pose_feat)
            pose_feat = self.pose_norm(pose_feat)
            video_feat = self.img_conv(video_feat)
            video_feat = self.img_norm(video_feat)
            
            pose_feat = pose_feat * pose_gate
            video_feat = video_feat * (1 - pose_gate)
        else:
            pose_gate = None
            pose_feat = self.pose_conv(pose_feat)
            video_feat = self.img_conv(video_feat)

        # aggregation
        feat = video_feat + pose_feat
        return feat, pose_gate


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        ## aggregation layer
        self.video_2_pose = make_convtrans3d_layers([self.img_resnet_dim, self.pose_resnet_dim], kernel=(1,1,2), stride=(1,1,2), padding=0)
        self.v2p_norm = nn.LayerNorm([self.pose_resnet_dim,  cfg.frame_per_seg , 7, 14])
        self.pose_2_video = make_conv3d_layers([self.pose_resnet_dim,self.img_resnet_dim], kernel=1, stride=(1,1,2),  padding=0)
        self.p2v_norm = nn.LayerNorm([self.img_resnet_dim,  cfg.frame_per_seg, 7, 7])


    def forward(self, video_feat, pose_feat):
        
        batch_size, video_feat_dim, video_feat_height, video_feat_width = video_feat.shape[0]//cfg.frame_per_seg, video_feat.shape[1],video_feat.shape[2], video_feat.shape[3]
        video_feat = video_feat.view(batch_size, cfg.frame_per_seg, video_feat_dim,  video_feat_height,  video_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0]//self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
        pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        
        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
                       pose_feat = self.pose_temporal_conv(pose_feat)
        
        v2p_feat = self.video_2_pose(video_feat)
        v2p_feat = self.v2p_norm(v2p_feat)
        #B, C, T, H, W
        p2v_feat = self.pose_2_video(pose_feat)
        p2v_feat = self.p2v_norm(p2v_feat)

        pose_feat = pose_feat + v2p_feat
        pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)
        video_feat = video_feat + p2v_feat
        video_feat = video_feat.permute(0,2,1,3,4).reshape(-1,video_feat_dim, video_feat_height, video_feat_width)
        #B*T, C, H, W
        pose_feat  = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]
        # aggregation
        feat = torch.cat((video_feat, pose_feat),1)
        # pose gate
        pose_gate = None
        return feat, pose_gate

class Aggregator_g(nn.Module):
    def __init__(self):
        super(Aggregator_g, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        #gated layer
        self.v2p_gate =make_convtrans3d_layers([self.img_resnet_dim, self.pose_resnet_dim], kernel=(1,1,2), stride=(1,1,2), padding=0,bnrelu_final=False)
        self.p2v_gate =make_conv3d_layers([self.pose_resnet_dim,self.img_resnet_dim], kernel=1, stride=(1,1,2),  padding=0,bnrelu_final=False)

        ## aggregation layer
        self.video_2_pose = make_convtrans3d_layers([self.img_resnet_dim, self.pose_resnet_dim], kernel=(1,1,2), stride=(1,1,2), padding=0)
        self.v2p_norm = nn.LayerNorm([self.pose_resnet_dim,  self.pose_frame_num , 7, 14])
        self.pose_2_video = make_conv3d_layers([self.pose_resnet_dim,self.img_resnet_dim], kernel=1, stride=(1,1,2),  padding=0)
        self.p2v_norm = nn.LayerNorm([self.img_resnet_dim,  cfg.frame_per_seg, 7, 7])


    def forward(self, video_feat, pose_feat):
        
        batch_size, video_feat_dim, video_feat_height, video_feat_width = video_feat.shape[0]//cfg.frame_per_seg, video_feat.shape[1],video_feat.shape[2], video_feat.shape[3]
        video_feat = video_feat.view(batch_size, cfg.frame_per_seg, video_feat_dim,  video_feat_height,  video_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0]//self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
        pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        
        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
                       pose_feat = self.pose_temporal_conv(pose_feat)
        
        v2p_feat = self.video_2_pose(video_feat)
        v2p_feat = self.v2p_norm(v2p_feat)
        v2p_gated = torch.sigmoid(self.v2p_gate(video_feat))
        #B, C, T, H, W
        p2v_feat = self.pose_2_video(pose_feat)
        p2v_feat = self.p2v_norm(p2v_feat)
        p2v_gated = torch.sigmoid(self.p2v_gate(pose_feat))

        pose_feat = pose_feat*v2p_gated + v2p_feat*(1-v2p_gated)
        pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)
        video_feat = video_feat*p2v_gated + p2v_feat*(1-p2v_gated)
        video_feat = video_feat.permute(0,2,1,3,4).reshape(-1,video_feat_dim, video_feat_height, video_feat_width)
        #B*T, C, H, W
        pose_feat  = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]
        # aggregation
        feat = torch.cat((video_feat, pose_feat),1)
        # pose gate
        pose_gate = None
        return feat, pose_gate

class Aggregator_s(nn.Module):
    def __init__(self):
        super(Aggregator_s, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        # co-encoder layer
        self.co_squeeze = make_conv_layers([self.img_resnet_dim+self.pose_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
  
        # excitation
        self.video_excitation = make_conv_layers([cfg.agg_feat_dim ,self.img_resnet_dim], kernel=1, padding=0, bnrelu_final=False)
        self.video_norm = nn.LayerNorm([cfg.agg_feat_dim, cfg.frame_per_seg, 1])

        self.pose_excitation = make_conv_layers([cfg.agg_feat_dim ,self.pose_resnet_dim], kernel=1, padding=0, bnrelu_final=False)
        self.pose_norm = nn.LayerNorm([cfg.agg_feat_dim, self.pose_frame_num, 1])

    def forward(self, video_feat, pose_feat):
        
        batch_size, video_feat_dim, video_feat_height, video_feat_width = video_feat.shape[0]//cfg.frame_per_seg, video_feat.shape[1],video_feat.shape[2], video_feat.shape[3]
        video_feat = video_feat.view(batch_size, cfg.frame_per_seg, video_feat_dim,  video_feat_height,  video_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0]//self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
        pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        
        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
                       pose_feat = self.pose_temporal_conv(pose_feat)
        
        pose_feat_  = pose_feat.mean((3,4))[:,:,:,None]
        video_feat_ = video_feat.mean((3,4))[:,:,:,None]
        #
        # aggregation
        co_feat = torch.cat((video_feat_, pose_feat_),1)
        co_feat = self.co_squeeze(co_feat)

        video_scale = torch.sigmoid(self.video_norm(self.video_excitation(co_feat)))[:,:,:,:, None]
        pose_scale = torch.sigmoid(self.pose_norm(self.pose_excitation(co_feat)))[:,:,:,:, None]
        #B, C, T, 1,1
        video_feat = video_feat*video_scale
        pose_feat = pose_feat*pose_scale

        video_feat = video_feat.permute(0,2,1,3,4).reshape(-1,video_feat_dim, video_feat_height, video_feat_width)
        pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)

        #B*T, C, H, W
        pose_feat  = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]
        # aggregation
        feat = torch.cat((video_feat, pose_feat),1)
        # pose gate
        pose_gate = None
        return feat, pose_gate

class Aggregator_f(nn.Module):
    def __init__(self):
        super(Aggregator_f, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]

        ## temporal strided conv to fuse with RGB
        self.pose_frame_num = (cfg.frame_per_seg-1) * cfg.pose_frame_factor + 1
        self.pose_temporal_conv = make_conv3d_layers([self.pose_resnet_dim, self.pose_resnet_dim], kernel=(5,1,1), stride=(cfg.pose_frame_factor,1,1), padding=(2,0,0))

        # co-encoder layer
        self.co_squeeze = make_conv_layers([self.img_resnet_dim+self.pose_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
  
        # excitation
        self.video_excitation = make_conv_layers([cfg.agg_feat_dim ,self.img_resnet_dim], kernel=1, padding=0, bnrelu_final=False)
        self.video_norm = nn.LayerNorm([cfg.agg_feat_dim, cfg.frame_per_seg, 1])

        self.pose_excitation = make_conv_layers([cfg.agg_feat_dim ,self.pose_resnet_dim], kernel=1, padding=0, bnrelu_final=False)
        self.pose_norm = nn.LayerNorm([cfg.agg_feat_dim, self.pose_frame_num, 1])
        #gated layer
        self.v2p_gate =make_convtrans_layers([cfg.frame_per_seg, self.pose_frame_num], kernel=(1,2), stride=(1,2), padding=0,bnrelu_final=False)
        self.p2v_gate =make_conv_layers([self.pose_frame_num,cfg.frame_per_seg], kernel=1, stride=(1,2),  padding=0,bnrelu_final=False)

        ## aggregation layer
        self.video_2_pose =make_convtrans_layers([cfg.frame_per_seg, self.pose_frame_num], kernel=(1,2), stride=(1,2), padding=0)
        self.v2p_norm = nn.LayerNorm([self.pose_frame_num, 7, 14])
        self.pose_2_video = make_conv_layers([self.pose_frame_num,cfg.frame_per_seg], kernel=1, stride=(1,2),  padding=0)
        self.p2v_norm = nn.LayerNorm([cfg.frame_per_seg, 7, 7])


    def forward(self, video_feat, pose_feat):
        
        batch_size, video_feat_dim, video_feat_height, video_feat_width = video_feat.shape[0]//cfg.frame_per_seg, video_feat.shape[1],video_feat.shape[2], video_feat.shape[3]
        video_feat = video_feat.view(batch_size, cfg.frame_per_seg, video_feat_dim,  video_feat_height,  video_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        batch_size, pose_feat_dim, pose_feat_height, pose_feat_width = pose_feat.shape[0]//self.pose_frame_num, pose_feat.shape[1], pose_feat.shape[2], pose_feat.shape[3]
        pose_feat = pose_feat.view(batch_size, self.pose_frame_num, pose_feat_dim, pose_feat_height, pose_feat_width).permute(0,2,1,3,4)
        #B, C, T, H, W
        
        # temporal fusing with RGB
        if cfg.pose_frame_factor > 1:
                       pose_feat = self.pose_temporal_conv(pose_feat)
        
        pose_feat_  = pose_feat.mean((3,4))[:,:,:,None]
        video_feat_ = video_feat.mean((3,4))[:,:,:,None]
        #
        # aggregation
        co_feat = torch.cat((video_feat_, pose_feat_),1)
        co_feat = self.co_squeeze(co_feat)

        video_scale = torch.sigmoid(self.video_norm(self.video_excitation(co_feat)))[:,:,:,:, None]
        pose_scale = torch.sigmoid(self.pose_norm(self.pose_excitation(co_feat)))[:,:,:,:, None]
        #B, C, T, 1,1
        video_feat = video_feat*video_scale
        pose_feat = pose_feat*pose_scale

        video_feat_p = video_feat.mean(1)
        pose_feat_p = pose_feat.mean(1)
        #B,T, H, W

        #cross gated
        v2p_feat = self.video_2_pose(video_feat_p)
        v2p_feat = self.v2p_norm(v2p_feat)[:,None,:,:,:]
        v2p_gated = torch.sigmoid(self.v2p_gate(video_feat_p))[:,None,:,:,:]
        #B, C, T, H, W
        p2v_feat = self.pose_2_video(pose_feat_p)
        p2v_feat = self.p2v_norm(p2v_feat)[:,None,:,:,:]
        p2v_gated = torch.sigmoid(self.p2v_gate(pose_feat_p))[:,None,:,:,:]

        pose_feat = pose_feat*v2p_gated + v2p_feat*(1-v2p_gated)
        pose_feat = pose_feat.permute(0,2,1,3,4).reshape(-1,pose_feat_dim, pose_feat_height, pose_feat_width)
        video_feat = video_feat*p2v_gated + p2v_feat*(1-p2v_gated)
        video_feat = video_feat.permute(0,2,1,3,4).reshape(-1,video_feat_dim, video_feat_height, video_feat_width)
        #B*T, C, H, W
        pose_feat  = pose_feat.mean((2,3))[:,:,None,None]
        video_feat = video_feat.mean((2,3))[:,:,None,None]
        # aggregation
        feat = torch.cat((video_feat, pose_feat),1)
        # pose gate
        
        return feat, v2p_gated, p2v_gated


class Classifier(nn.Module):
    def __init__(self, class_num):
        super(Classifier, self).__init__()
        self.class_num = class_num
        if cfg.mode == 'rgb_only':
            self.fc = make_linear_layers([cfg.resnet_feat_dim[cfg.img_resnet_type], self.class_num], relu_final=False)
        elif cfg.mode == 'pose_only':
            self.fc = make_linear_layers([cfg.resnet_feat_dim[cfg.pose_resnet_type], self.class_num], relu_final=False)
        else:
            self.fc = make_linear_layers([cfg.resnet_feat_dim[cfg.img_resnet_type]+cfg.resnet_feat_dim[cfg.pose_resnet_type], self.class_num], relu_final=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, video_feat):
        video_feat = video_feat.mean((2,3))
        label_out = self.dropout(video_feat) # dropout
        label_out = self.fc(video_feat)
        return label_out

