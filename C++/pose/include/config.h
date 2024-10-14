#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>


const int kGpuId = 0;
const int kNumClass = 1;
const int kNumKpt = 17;  // 单个目标对应的关键点的个数
const int kKptDims = 3;  // 单个关键点的维度，2 for x,y or 3 for x,y,visible
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
const int kNumBoxElement = 7 + kNumKpt * kKptDims;  // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS), 51 keypoints

const std::string onnxFile = "../onnx_model/yolo11s-pose.onnx";
// const std::string trtFile = "./yolo11s.plan";
// const std::string testDataDir = "../images";  // 用于推理

// for FP16 mode
const bool bFP16Mode = false;
// for INT8 mode
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";  // 存放用于 int8 量化校准的图像

const std::vector<std::string> vClassNames {"person"};

const std::vector<std::vector<int>> skeleton {
    {16, 14},
    {14, 12},
    {17, 15},
    {15, 13},
    {12, 13},
    {6, 12},
    {7, 13},
    {6, 7},
    {6, 8},
    {7, 9},
    {8, 10},
    {9, 11},
    {2, 3},
    {1, 2},
    {1, 3},
    {2, 4},
    {3, 5},
    {4, 6},
    {5, 7}
};

#endif  // CONFIG_H
