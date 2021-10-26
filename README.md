注意！！！！！！！
这个仓库只是把原stgcn的openpose接口替换为light_openpose，没有对其它代码进行改动，如果你只是训练和测试那么你并不需要任何openpose接口。

我没有在原demo_offline上进行改动，而是重新整合了一个draft.py文件，如果你想输入视频，那么修改116行的路径为你的路径。如果你想调用摄像头，修改117行。

你需要下载light_openpose的预训练模型(https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)，   然后修改104行为你的路径。

(请忽略在运行过程中出现的一切输出，那是我在测试时候写的，懒得删了(诶嘿))
有问题请提出。



以下为原readme！！！！！
## Reminder

ST-GCN has transferred to [MMSkeleton](https://github.com/open-mmlab/mmskeleton),
and keep on developing as an flexible open source toolbox for skeleton-based human understanding.
You are welcome to migrate to new MMSkeleton.
Custom networks, data loaders and checkpoints of old st-gcn are compatible with MMSkeleton.
If you want to use old ST-GCN, please refer to [OLD_README.md](./OLD_README.md).

This code base will soon be not maintained and exists as a historical artifact to supplement our AAAI papers on:

> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

For more recent works please checkout MMSkeleton.
  
