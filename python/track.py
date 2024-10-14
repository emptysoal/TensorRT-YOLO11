# -*- coding: utf-8 -*-

import os
import time
import argparse

import cv2
import numpy as np

from detect.infer import YoloDetector
from tracker.byte_tracker import BYTETracker

# 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
track_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


def make_parser():
    parser = argparse.ArgumentParser("YOLOv8 ByteTrack.")
    parser.add_argument("--detect-model",
                        type=str,
                        default="./detect/model.plan",
                        help="Tensorrt plan path of YOLOv8 detection.")
    parser.add_argument("--video",
                        type=str,
                        default="./videos/street.mp4",
                        help="The path of the video to be tracked.")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def main(args):
    assert os.path.isfile(args.video), "Video path does not exist."

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames: %s" % n_frames)

    vid_writer = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    # YOLOv8 predictor
    yolo_infer = YoloDetector(trt_plan=args.detect_model, gpu_id=0, nms_thresh=0.45, conf_thresh=0.01)

    # ByteTrack tracker
    tracker = BYTETracker(args, frame_rate=30)

    num_frames = 0
    total_cost = 0

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break
        num_frames += 1
        if num_frames % 20 == 0:
            print("Processing frame : %s ( %s fps)" % (num_frames, int(num_frames / total_cost)))

        start = time.time()

        # YOLOv8 inference
        detect_res = yolo_infer.inference(frame)

        # 筛选出想要跟踪的类别
        classes = (detect_res[:, 5]).astype(np.int32)
        valid = np.isin(classes, track_classes)
        track_input = detect_res[valid]

        # bytetrack track.
        online_targets = tracker.update(track_input)

        end = time.time()
        total_cost += end - start

        # draw track result
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            color = get_color(t.track_id)
            cv2.putText(frame, str(int(t.track_id)), (int(x1), int(y1) - 3), 0, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=color, thickness=2)
        cv2.putText(frame,
                    'frame: %d fps: %d num: %d' % (num_frames, int(num_frames / total_cost), len(online_targets)),
                    (0, 30), 0, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        vid_writer.write(frame)

        # cv2.imshow("Frame", frame)
        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break

    yolo_infer.release()
    cap.release()


if __name__ == '__main__':
    flags = make_parser().parse_args()
    main(flags)
