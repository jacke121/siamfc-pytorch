from __future__ import division
import sys
import os

import cv2
import numpy as np
from PIL import Image
# import src.siamese as siam
from src.parse_arguments import parse_arguments

import numpy
import numpy as np


def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return cx, cy, w, h
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx.astype(numpy.int64), cy.astype(numpy.int64), w.astype(numpy.int64), h.astype(numpy.int64)
    else:
        return (cx - w / 2).astype('float32'), (cy - h / 2).astype('float32'), w.astype('float32'), h.astype('float32')


def main():
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # siam = nn.DataParallel(siam)
    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'bag':
        dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
        videos_list = [v for v in os.listdir(dataset_folder) if not v[0] == '.']
        videos_list.sort()
        nv = np.size(videos_list)
        speed = np.zeros(nv * evaluation.n_subseq)
        precisions = np.zeros(nv * evaluation.n_subseq)
        precisions_auc = np.zeros(nv * evaluation.n_subseq)
        ious = np.zeros(nv * evaluation.n_subseq)
        lengths = np.zeros(nv * evaluation.n_subseq)
        for i in range(nv):
            print('video: %d' % (i + 1))
            gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[i])
            starts = np.rint(np.linspace(0, n_frames - 1, evaluation.n_subseq + 1))
            starts = starts[0:evaluation.n_subseq]
            for j in range(evaluation.n_subseq):
                start_frame = int(starts[j])
                gt_ = gt[start_frame:, :]
                frame_name_list_ = frame_name_list[start_frame:]
                pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])

                img=cv2.imread(frame_name_list_[0])
                pos_x=pos_x * 0.85
                pos_y=pos_y*0.8
                cv2.rectangle(img,(int(pos_x),int(pos_y)),(int(pos_x+target_w), int(pos_y+target_h)),(0,255,0),2)

                cv2.imshow("asadf",img)
                cv2.waitKeyEx()





def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames

if __name__ == '__main__':
    sys.exit(main())
