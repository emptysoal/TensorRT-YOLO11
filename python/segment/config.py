# -*- coding:utf-8 -*-

kGpuId = 0
kNumClass = 80
kInputH = 640
kInputW = 640
kNmsThresh = 0.45
kConfThresh = 0.25

onnx_file = "./onnx_model/yolo11s-seg.onnx"
trt_file = "./model.plan"

# for FP16 mode
use_fp16_mode = False
# for INT8 mode
use_int8_mode = False
n_calibration = 20
cache_file = "./int8.cache"
calibration_data_dir = "./calibrator"  # 存放用于 int8 量化校准的图像

class_name_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
