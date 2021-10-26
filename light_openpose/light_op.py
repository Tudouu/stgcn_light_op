import cv2
import numpy as np
import torch
from time import *
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.keypoints import extract_keypoints, group_keypoints
from light_openpose.modules.load_state import load_state
from light_openpose.modules.pose import Pose, track_poses
from val import normalize, pad_width

net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu=False,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):

    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)#这行下一行是10**-2级别
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]#这行下一行是10**-4级别
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    #if not cpu:
    tensor_img = tensor_img.cuda()#这行下一行是10**-5级别
    net = net.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad



def estimate_pose(net,img):
    fill=[]
    net = net.eval()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    heatmaps, pafs, scale, pad = infer_fast(net, img,256, stride, upsample_ratio, True)
    total_keypoints_num = 0
    all_keypoints_by_type = []

    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    if len(pose_entries)==0:
        return np.zeros((18,3))
    #print('pose_entries',pose_entries)
    #print('all_keypoints',all_keypoints)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 3), dtype=np.float) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose_keypoints[kpt_id, 2]=float(all_keypoints[int(pose_entries[n][kpt_id]), 2])
        #print('pose_keypoints',pose_keypoints)
        pose = Pose(pose_keypoints, pose_entries[n][18])#21/9
        return pose.bbox
        #print('pose.bbox',pose.bbox)
        #for i in range(18):
            #cv2.circle(orig_img,(157,176),6,(255,0,0),5)
        #cv2.imshow('pic',orig_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

