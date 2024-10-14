# -*- coding:utf-8 -*-

"""
    YOLO 图像预处理
"""

import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img


def preprocess(np_img, dst_height, dst_width):
    img = letterbox(np_img, (dst_height, dst_width))

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = img / 255.0

    img = img.astype(np.float32)
    img = np.ascontiguousarray(img)

    return img


if __name__ == '__main__':
    path = "./images/zidane.jpg"
    img0 = cv2.imread(path, 1)
    img_out = letterbox(img0, (640, 640))
    cv2.imshow("letterbox", img_out)
    cv2.waitKey()
