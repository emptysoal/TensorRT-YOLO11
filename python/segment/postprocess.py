# -*- coding:utf-8 -*-

"""
    YOLOv8 图像预处理
"""

import numpy as np
import cv2


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


def scale_masks(masks, img0_shape):
    """
        Takes a mask, and resizes it to the original image size.
    :param masks: (np.ndarray) resized and padded masks, [num, h, w].
    :param img0_shape: (tuple) the original image shape
    :return: The masks that are being returned.
    """
    masks = masks.transpose((1, 2, 0))  # CHW to HWC, (n, 160, 160) to (160, 160, n)
    img1_shape = masks.shape  # (160, 160, n)
    if img1_shape[:2] == img0_shape[:2]:
        masks = masks.transpose((2, 0, 1))  # HWC to CHW
        return masks
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    masks = masks.transpose((2, 0, 1))  # HWC to CHW

    mask1 = masks > 0.5
    mask2 = masks <= 0.5
    masks[mask1] = 1
    masks[mask2] = 0

    return masks


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def crop_mask(masks, boxes):
    """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
    :param masks: [n, h, w] ndarray of masks
    :param boxes: [n, 4] ndarray of bbox coordinates in relative point form
    :return: The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1 = boxes[:, 0][:, None, None]
    y1 = boxes[:, 1][:, None, None]
    x2 = boxes[:, 2][:, None, None]
    y2 = boxes[:, 3][:, None, None]

    c = np.arange(w, dtype=x1.dtype)[None, None, :]  # cols shape(1,1,w)
    r = np.arange(h, dtype=x1.dtype)[None, :, None]  # rows shape(1,h,1)

    return masks * ((c >= x1) * (c < x2) * (r >= y1) * (r < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
        Apply masks to bounding boxes using the output of the mask head.
    :param protos: A ndarray of shape [mask_dim, mask_h, mask_w], default [32, 160, 160].
    :param masks_in: A ndarray of shape [n, mask_dim], where n is the number of masks after NMS.
    :param bboxes: A ndarray of shape [n, 4], where n is the number of masks after NMS.
    :param shape: A tuple of integers representing the size of the input image in the format (h, w).
    :param upsample: A flag to indicate whether to upsample the mask to the original image size. Default is False.
    :return:
        (ndarray): A binary mask ndarray of shape [n, h, w], where n is the number of masks after NMS, and h and w
        are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # 32, 160, 160
    ih, iw = shape  # 640, 640
    masks = sigmoid(masks_in @ protos.reshape((c, -1))).reshape((-1, mh, mw))  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 1] *= height_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = masks.transpose((1, 2, 0))  # CHW to HWC
        masks = cv2.resize(masks, (iw, ih), interpolation=cv2.INTER_LINEAR)  # bilinear upsample
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        masks = masks.transpose((2, 0, 1))  # HWC to CHW

    return masks


def postprocess(img0, preds, conf_thres, iou_thres, input_h, input_w, nc=0):
    """
    img0: original image
    preds: YOLOv8 segment output, tuple (out1, out2)
                out1: shape (32, 160, 160)   out2: shape (116, 8400)
    """
    prediction = preds[1]  # shape (116, 8400)

    nc = nc or (prediction.shape[0] - 4)  # number of classes
    nm = prediction.shape[0] - nc - 4  # number of mask coefficient, default 32
    mi = 4 + nc  # mask start index
    xc = prediction[4:mi].max(0) > conf_thres  # [ True, False,  True, False,  True,  True, ... ]

    prediction = prediction.transpose((1, 0))
    x = prediction[xc]

    if not x.shape[0]:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 640))

    box = x[:, :4]
    cls = x[:, 4:mi]
    mask = x[:, mi:]

    i, j = np.where(cls > conf_thres)
    x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None], mask[i]), 1)
    # x shape : (num_bboxes, 38)
    # 38 dims are : center_x, center_y, w, h, conf, class_id, 32 mask
    bboxes = xywh2xyxy(x)
    labels = set(bboxes[:, 5].astype(int))

    detected_objects = []  # [[x1, y1, x2, y2, conf, class_id, 32 mask], [...], [...]]
    for label in labels:
        selected_bboxes = bboxes[np.where(bboxes[:, 5] == label)]
        selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], iou_thres)]
        detected_objects += selected_bboxes_keep.tolist()

    if detected_objects:
        detected_objects = np.array(detected_objects)
    else:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 640))

    proto = preds[0]  # shape (32, 160, 160)
    masks = process_mask(proto, detected_objects[:, 6:], detected_objects[:, :4], (input_h, input_w))
    # masks: shape (num_bboxes, input_h / 4, input_w / 4), such as (8, 160, 160)
    masks = scale_masks(masks, img0.shape)
    # masks: shape (num_bboxes, origin_h, origin_w), such as (8, 476, 538)

    detected_objects[:, :4] = scale_coords((input_h, input_w), detected_objects[:, :4], img0.shape[:2]).round()

    return detected_objects[:, :6], masks
