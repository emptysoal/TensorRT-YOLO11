#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include "config.h"

using namespace nvinfer1;


class YoloDetector
{
public:
    YoloDetector(const std::string trtFile);
    ~YoloDetector();
    std::vector<Detection> inference(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox=true);

private:
    void get_engine();
    static void process_mask(
        float* protoDevice, Dims32 protoOutDims, std::vector<Detection>& vDetections, 
        int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
    );

private:
    Logger              gLogger;
    std::string         trtFile_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevide;
    float *             decodeDevice;

    int                 OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
    Dims32              protoOutDims;  // proto shape [1 32 160 160]
};

#endif  // INFER_H
