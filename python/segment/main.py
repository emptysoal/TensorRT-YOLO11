# -*- coding:utf-8 -*-

import os
import time
import cv2

from infer import YoloDetector

load_start = time.time()

# instance
yolo_infer = YoloDetector(trt_plan="./model.plan", gpu_id=0)

load_end = time.time()
print("Model load cost: %.4f s" % (load_end - load_start))

for img_name in os.listdir("./images"):
    img_path = os.path.join("./images", img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    infer_start = time.time()

    bboxes, masks = yolo_infer.inference(img)

    infer_end = time.time()
    print("Infer image %s cost %d ms." % (img_name, (infer_end - infer_start) * 1000))

    YoloDetector.draw_image(img, bboxes, masks)

    save_dir = "./output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir + "_" + img_name, img)

yolo_infer.release()
