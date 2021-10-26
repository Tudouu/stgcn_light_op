import torch
import cv2
from processor.demo_offline import naive_pose_tracker
#from openpose.src.body import Body
import numpy as np
import torch
import sys
import traceback
from collections import OrderedDict
import tools.utils as utils
from time import *
from light_openpose import light_op
from light_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from light_openpose.modules.load_state import load_state


#file_dir='/media/wow/disk2/AG/st-gcn-master/models/st_gcn.kinetics.pt'
#model=torch.load(file_dir)
#print(model)
#picture='/media/wow/disk2/AG/dataset/frames/V95RI.mp4/000466.png'
#picture='/media/wow/disk2/AG/test.mp4'
#pic=cv2.VideoCapture(picture)
#length=pic.get(cv2.CAP_PROP_FRAME_COUNT)
#print(length)

def load_model(model, **model_args):
    # model=GCN.st_gcn.Model
    Model = import_class(model)
    model = Model(**model_args)
    return model


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def load_weights(model, weights, ignore_weights):
    model = load(model, weights, ignore_weights).cuda()
    return model


def load(model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        state.update(weights)
        model.load_state_dict(state)
    return model


def estimate_bodypose(candidate, subset):
    #candidate: x, y, score, id
    #[[-1.     0.     1.     2.    - 1.     3.     4.     5.     6.     7.     8.     9.    10.    11.    - 1.    - 1.    12.    13.    22.854   14.]]
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]#19
    yet={'0':'false','1':'false','2':'false','3':'false','4':'false','5':'false','6':'false','7':'false',
              '8':'false','9':'false','10':'false','11':'false','12':'false','13':'false','14':'false',
              '15':'false','16':'false','17':'false'}
    posed_index=[]
    posed_index_not=[]
    posed_value=[]
    for i in range(18):#i为关节点
        index = int(subset[0][i])#行
        if index == -1:
            posed_index_not.append(i)
            posed_value.append([0,0,0])
            continue
        else:
            posed_index.append(i)
            posed_value.append(candidate[index][0:3])#x, y, score

    posed_value=np.array(posed_value)

    return posed_value,posed_index,posed_index_not

def render_video(data_numpy, voting_label_name, video_label_name, intensity, video):
    images = utils.visualization.stgcn_visualize(
        data_numpy,
        model.graph.edge,
        intensity, video,
        voting_label_name,
        video_label_name,
        339)
    return images

net = PoseEstimationWithMobileNet()
checkpoint = torch.load('/media/wow/disk2/STT2/STTran-main/light_openpose/checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)

label_name_path = './resource/kinetics_skeleton/label_name.txt'
with open(label_name_path) as f:
    label_name = f.readlines()
    label_name = [line.rstrip() for line in label_name]
    label_name = label_name

#body_estimate=Body("openpose/model/body_pose_model.pth")

video_path='/media/wow/disk2/AG/st-gcn-master/resource/media/clean_and_jerk.mp4'
videos=cv2.VideoCapture(video_path)
length=videos.get(cv2.CAP_PROP_FRAME_COUNT)
time_a=time()
pose_tracker = naive_pose_tracker(data_frame=length)
frame_index = 0
i=0
video=list()
a1=time()
aaa=0
while (True):
    # get image
    at = time()
    ret, orig_image = videos.read()
    aaa=aaa+1

    if orig_image is None:
        break
    source_H, source_W, _ = orig_image.shape
    orig_image = cv2.resize(
        orig_image, (256 * source_W // source_H, 256))
    H, W, _ = orig_image.shape
    video.append(orig_image)
    #candidate,subset=body_estimate(orig_image)

    #multi_pose,posed_index,posed_index_not= estimate_bodypose(candidate, subset)
    multi_pose = light_op.estimate_pose(net,orig_image)
    #cv2.circle(orig_image,(73,71),1,(255,0,0),-1)
    #cv2.imshow('image',orig_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    multi_pose=torch.from_numpy(multi_pose)
    multi_pose=multi_pose.unsqueeze(0)
    multi_pose[:, :, 0] = multi_pose[:, :, 0] / W  # 横坐标
    multi_pose[:, :, 1] = multi_pose[:, :, 1] / H  # 纵坐标
    multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
    multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
    multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

    multi_pose=multi_pose.numpy()
    pose_tracker.update(multi_pose, frame_index)
    frame_index += 1
    i=i+1
   # print('Pose estimation ({}/{}).'.format(frame_index))
   # if frame_index==2:
   #     break
a2 = time()

print('阶段1时间{}'.format(a2 - a1))#41.52311873435974
data_numpy = pose_tracker.get_skeleton_sequence()
weight_Path = '/media/wow/disk2/STT2/STTran-main/GCN/st_gcn.kinetics.pt'
first_para = 'net.st_gcn.Model'
second_para = {'in_channels': 3, 'num_class': 400, 'edge_importance_weighting': True,
               'graph_args': {'layout': 'openpose', 'strategy': 'spatial'}}
data = torch.from_numpy(data_numpy)
data = data.unsqueeze(0)
data = data.float().to("cuda:0").detach()
#print(data.shape)[1, 3, 339, 18, 1]

model = load_model(first_para, **second_para)
model = load_weights(model, weight_Path, None)
output, feature=model.extract_feature(data)
time_c=time()

output = output[0]
#print(output.shape)[400, 85, 18, 1]
feature = feature[0]
#print(feature.shape)[256, 85, 18, 1]
intensity = (feature * feature).sum(dim=0) ** 0.5
#print(intensity.shape)[85, 18, 1]
intensity = intensity.cpu().detach().numpy()

# get result
# classification result of the full sequence
voting_label = output.sum(dim=3).sum(
    dim=2).sum(dim=1).argmax(dim=0)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',voting_label)
voting_label_name=label_name[voting_label]
print(voting_label_name)
# classification result for each person of the latest frame
num_person = data.size(4)
latest_frame_label = [output[:, :, :, m].sum(
    dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
latest_frame_label_name = [label_name[l]
                           for l in latest_frame_label]
print(latest_frame_label_name)
num_person = output.size(3)
num_frame = output.size(1)
video_label_name = list()
for t in range(num_frame):
    frame_label_name = list()
    for m in range(num_person):
        person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
        person_label_name = label_name[person_label]
        frame_label_name.append(person_label_name)
    video_label_name.append(frame_label_name)
print('+-------------------------------------------++++++++++++++++++++-',len(video_label_name))
print(video_label_name)

print('----------------------------')
print(np.shape(video_label_name))



images = render_video(data_numpy, voting_label_name,
                           video_label_name, intensity, video)
numm=0
for image in images:
    numm=numm+1
    image = image.astype(np.uint8)
    cv2.putText(image,str(numm),(12,25),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),3)
    cv2.imshow("ST-GCN", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break