import cv2
import numpy as np

from light_openpose.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from light_openpose.modules.one_euro_filter import OneEuroFilter


class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.fill=[]
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        #(18,3)
        #[[1.05000000e+02  2.06000000e+02  3.04701596e-01]
         #[9.70000000e+01  2.28000000e+02  4.12792891e-01]
        #[8.20000000e+01 2.21000000e+02 5.24159849e-01]
        #[6.00000000e+01  2.58000000e+02  4.74704385e-01]
        #[6.00000000e+01 3.00000000e+02 3.18513244e-01]
        #[1.20000000e+02  2.28000000e+02  3.58960003e-01]
        #[-1.00000000e+00 - 1.00000000e+00 - 1.00000000e+00]
        #[-1.00000000e+00 - 1.00000000e+00 - 1.00000000e+00]
        #[7.50000000e+01 3.00000000e+02 4.77475166e-01]
        #[7.50000000e+01  3.48000000e+02  3.72693896e-01]
        #[6.00000000e+01 4.05000000e+02 3.32490206e-01]
        #[1.01000000e+02  3.00000000e+02  2.95110673e-01]
        #[9.30000000e+01 3.52000000e+02 1.76126942e-01]
        #[9.00000000e+01  4.01000000e+02  1.84243828e-01]
        #[1.01000000e+02 1.98000000e+02 2.86982417e-01]
        #[1.08000000e+02  2.02000000e+02  2.27350965e-01]
        #[9.30000000e+01 2.02000000e+02 3.65057558e-01]
        #[-1.00000000e+00 - 1.00000000e+00 - 1.00000000e+00]]
        found_keypoints = np.zeros((18, 3), dtype=np.float)#(18,3)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):#kpt_id=0,1,2,3...
            if keypoints[kpt_id, 0] == -1:
                found_keypoints[found_kpt_id]=[0,0,0]
                found_kpt_id += 1
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        #bbox = cv2.boundingRect(found_keypoints)
        for i in range(18):
            for j in range(2):
                found_keypoints[i][j] = int(found_keypoints[i][j])
        return found_keypoints

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
