import torch
import numpy as np
import cv2
from openpose.src.body import Body
from openpose.src.estimate_pose import estimate_bodypose
import torchlight
from GCN.gcn import Model
import collections
from time import *
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.load_state import load_state
from light_openpose import light_op



root_path='/media/wow/disk2/AG/dataset/frames/'
net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth')
#checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)






def transform(gt_pose_before):

    gt_pose=np.zeros((18,3),float)

    #print(gt_pose_before[0]['pose_17'][0][6])
    gt_pose[0]=gt_pose_before[0]['pose_17'][0][0]
    gt_pose[2]=gt_pose_before[0]['pose_17'][0][6]
    gt_pose[3] = gt_pose_before[0]['pose_17'][0][8]
    gt_pose[4] = gt_pose_before[0]['pose_17'][0][10]
    gt_pose[5] = gt_pose_before[0]['pose_17'][0][5]
    gt_pose[6] = gt_pose_before[0]['pose_17'][0][7]
    gt_pose[7] = gt_pose_before[0]['pose_17'][0][9]
    gt_pose[8] = gt_pose_before[0]['pose_17'][0][12]
    gt_pose[9] = gt_pose_before[0]['pose_17'][0][14]
    gt_pose[10] = gt_pose_before[0]['pose_17'][0][16]
    gt_pose[11] = gt_pose_before[0]['pose_17'][0][11]
    gt_pose[12] = gt_pose_before[0]['pose_17'][0][13]
    gt_pose[13] = gt_pose_before[0]['pose_17'][0][15]
    gt_pose[14] = gt_pose_before[0]['pose_17'][0][2]
    gt_pose[15] = gt_pose_before[0]['pose_17'][0][1]
    gt_pose[16] = gt_pose_before[0]['pose_17'][0][4]
    gt_pose[17] = gt_pose_before[0]['pose_17'][0][3]
    gt_pose[1]=(gt_pose[2]+gt_pose[5])//2

    return gt_pose


def extract(mode, openpose_index, openpose_index_len, openposed_value_scale, openpose_rel_man_img_idx, gt_pose):
    i = 0
    all_pose = []

    if mode:
        fre_ske_idx = []
        openpose_rel_man_img_idx = np.array(openpose_rel_man_img_idx)
        while i < len(openpose_rel_man_img_idx):
            if i == len(openpose_rel_man_img_idx) - 1:
                fre_ske_idx.append(1)
                break
            num = 1
            while (i + 1) < len(openpose_rel_man_img_idx) and openpose_rel_man_img_idx[i] == openpose_rel_man_img_idx[
                i + 1]:
                num = num + 1
                if (i + 1) < len(openpose_rel_man_img_idx):
                    i = i + 1
                else:
                    fre_ske_idx.append(num)
                    break
            if i == len(openpose_rel_man_img_idx) - 1:
                fre_ske_idx.append(num)
                break
            fre_ske_idx.append(num)
            i = i + 1
        fre_ske_idx = np.array(fre_ske_idx)

        for i in range(openpose_index_len):
            # annotations:
            # pose = np.array([[193, 71, 1.],  # 鼻子 0
            #                 [198, 67, 1.],  # 左眼 1
            #                 [191, 67, 1.],  # 右眼 2
            #                 [211, 70, 1.],  # 左耳 3
            #                 [187, 71, 1.],  # 右耳 4
            #                 [229, 105, 1.],  # 左肩 5
            #                 [186, 103, 1.],  # 右肩 6
            #                 [238, 146, 1.],  # 左肘关节 7
            #                 [163, 123, 1.],  # 右肘关节 8
            #                 [236, 184, 1.],  # 左手腕 9
            #                 [176, 94, 1.],  # 右手腕 10
            #                 [220, 184, 1.],  # 左胯关节 11
            #                 [189, 184, 1.],  # 右胯关节 12
            #                 [221, 238, 1.],  # 左膝盖 13
            #                 [189, 240, 1.],  # 右膝盖 14
            #                 [224, 267, 1.],  # 左脚踝 15
            #                 [191, 267, 1.]]  # 右脚踝 16
            #                 [[(左肩+右肩)/2]]) #轴骨 17
            # print(root_path + openpose_index[i])
            # print(openposed_value_scale)
            gt_pose_before = gt_pose[i]
            image = cv2.imread(root_path + openpose_index[i])
            # openposed_value_scale大于1是放大，小于1是缩小
            #TODO 先resize成340*256?

            image = cv2.resize(image, None, None, fx=float(openposed_value_scale), fy=float(openposed_value_scale),
                               interpolation=cv2.INTER_LINEAR)
            w, h, _ = image.shape

            multi_pose = light_op.estimate_pose(net, image)
            gt_pose_after = transform(gt_pose_before)
            gt_pose_after[:, :2] = gt_pose_after[:, :2] * float(openposed_value_scale)
            # print(multi_pose)

            # if len(multi_pose)==0:
            # multi_pose=torch.ones((17,3))
            for exch in range(18):
                if multi_pose[exch][2] == 0:
                    multi_pose[exch] = gt_pose_after[exch]
            multi_pose = torch.from_numpy(multi_pose)
            # multi_pose = multi_pose.unsqueeze(0)
            multi_pose[:, 0] = multi_pose[:, 0] / w  # 横坐标
            multi_pose[:, 1] = multi_pose[:, 1] / h  # 纵坐标
            multi_pose[:, 0:2] = multi_pose[:, 0:2] - 0.5

            # for xh in range(fre_ske_idx[i]):
            all_pose.append(multi_pose)

        return all_pose, fre_ske_idx
    else:
        for i in range(openpose_index_len):
            # annotations:
            # pose = np.array([[193, 71, 1.],  # 鼻子 0
            #                 [198, 67, 1.],  # 左眼 1
            #                 [191, 67, 1.],  # 右眼 2
            #                 [211, 70, 1.],  # 左耳 3
            #                 [187, 71, 1.],  # 右耳 4
            #                 [229, 105, 1.],  # 左肩 5
            #                 [186, 103, 1.],  # 右肩 6
            #                 [238, 146, 1.],  # 左肘关节 7
            #                 [163, 123, 1.],  # 右肘关节 8
            #                 [236, 184, 1.],  # 左手腕 9
            #                 [176, 94, 1.],  # 右手腕 10
            #                 [220, 184, 1.],  # 左胯关节 11
            #                 [189, 184, 1.],  # 右胯关节 12
            #                 [221, 238, 1.],  # 左膝盖 13
            #                 [189, 240, 1.],  # 右膝盖 14
            #                 [224, 267, 1.],  # 左脚踝 15
            #                 [191, 267, 1.]]  # 右脚踝 16
            #                 [[(左肩+右肩)/2]]) #轴骨 17
            # print(root_path + openpose_index[i])
            # print(openposed_value_scale)
            image = cv2.imread(root_path + openpose_index[i])
            # openposed_value_scale大于1是放大，小于1是缩小
            image = cv2.resize(image, None, None, fx=float(openposed_value_scale), fy=float(openposed_value_scale),
                               interpolation=cv2.INTER_LINEAR)
            w, h, _ = image.shape

            multi_pose = light_op.estimate_pose(net, image)
            multi_pose[:, :2] = multi_pose[:, :2] * float(openposed_value_scale)
            # print(multi_pose)

            # if len(multi_pose)==0:
            # multi_pose=torch.ones((17,3))
            multi_pose = torch.from_numpy(multi_pose)
            # multi_pose = multi_pose.unsqueeze(0)
            multi_pose[:, 0] = multi_pose[:, 0] / w  # 横坐标
            multi_pose[:, 1] = multi_pose[:, 1] / h  # 纵坐标
            multi_pose[:, 0:2] = multi_pose[:, 0:2] - 0.5

            # for xh in range(fre_ske_idx[i]):
            all_pose.append(multi_pose)

        return all_pose