#ifndef DRAW_H
#define DRAW_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void draw_mask(cv::Mat& img, float* mask);

#endif  // DRAW_H
