# -*- coding:utf-8 -*-

kGpuId = 0
kNumClass = 1
kKptShape = [17, 3]  # [number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)]
kInputH = 640
kInputW = 640
kNmsThresh = 0.45
kConfThresh = 0.25

onnx_file = "./onnx_model/yolo11s-pose.onnx"
trt_file = "./model.plan"

# for FP16 mode
use_fp16_mode = False
# for INT8 mode
use_int8_mode = False
n_calibration = 20
cache_file = "./int8.cache"
calibration_data_dir = "./calibrator"  # 存放用于 int8 量化校准的图像

class_name_list = ["person"]

# 人体关键点检测时，关键点间骨骼连接信息
skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]
