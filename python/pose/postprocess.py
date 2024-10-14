# -*- coding:utf-8 -*-

"""
    YOLOv8 图像预处理
"""

import numpy as np


def xywh2xyxy(bboxes):
    """
        Convert nx4 boxes from [center x, center y, w, h, conf, class_id] to [x1, y1, x2, y2, conf, class_id]
    """
    out = bboxes.copy()
    out[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
    out[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
    out[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # bottom right x
    out[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # bottom right y
    return out


def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        indexes = np.where(iou <= threshold)[0]
        order = order[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_kpt_coords(img1_shape, coords, img0_shape):
    # Rescale key points coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[..., 0] -= pad[0]  # x padding
    coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain

    return coords


def postprocess(img0, prediction, conf_thres, iou_thres, input_h, input_w, nc=0, kpt_shape=(17, 3)):
    """
    img0: original image
    prediction: YOLOv8 pose output and after reshape, pose default shape is [56, 8400], 56 = 4 + 1 + 51
    """
    nc = nc or (prediction.shape[0] - 4)
    nm = prediction.shape[0] - nc - 4
    mi = 4 + nc  # keypoint start index
    xc = prediction[4:mi].max(0) > conf_thres  # [ True, False,  True, False,  True,  True, ... ]

    prediction = prediction.transpose((1, 0))
    x = prediction[xc]

    if not x.shape[0]:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 17, 3))

    box = x[:, :4]
    cls = x[:, 4:mi]
    kpt = x[:, mi:]

    i, j = np.where(cls > conf_thres)
    x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None], kpt[i]), 1)
    # x shape : (num_bboxes, 57)
    # 57 dims are : center_x, center_y, w, h, conf, class_id, 51 * keypoints
    bboxes = xywh2xyxy(x)
    labels = set(bboxes[:, 5].astype(int))

    detected_objects = []  # [[x1, y1, x2, y2, conf, class_id, 51 kpts], [...], [...]]
    for label in labels:
        selected_bboxes = bboxes[np.where(bboxes[:, 5] == label)]
        selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], iou_thres)]
        detected_objects += selected_bboxes_keep.tolist()

    if detected_objects:
        detected_objects = np.array(detected_objects)
    else:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 17, 3))

    detected_objects[:, :4] = scale_coords((input_h, input_w), detected_objects[:, :4], img0.shape[:2]).round()

    pred_kpts = detected_objects[:, 6:].reshape((len(detected_objects), *kpt_shape))
    pred_kpts = scale_kpt_coords((input_h, input_w), pred_kpts, img0.shape[:2])

    return detected_objects[:, :6], pred_kpts
