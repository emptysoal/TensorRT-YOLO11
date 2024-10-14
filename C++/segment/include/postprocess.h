#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);
/*
    transpose [116 8400] convert to [8400 116]
src:          Tensor, dim is [116 8400]
dst:          Tensor, dim is [8400 116]
numBboxes:    number of bboxes: default 8400
numElements:  center_x, center_y, width, height, 80 classes, 32 masks
*/

void decode(float* src, float* dst, int numBboxes, int numClasses, int numMasks, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);
/*
    convert [8400 116] to [39001, ], 39001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 32masks), 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 4bbox: left, top, right, bottom)
*/

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);


void matrix_multiply(float* aMatrix, int aRows, int aCols, float* bMatrix, int bRows, int bCols, float* cMatrix, cudaStream_t stream, bool sigm = false);
/*
    matrix multiply, like numpy.matmul() function
aMatrix:          input matrix 1 array on device
aRows:            rows of input matrix 1
aCols:            columns of input matrix 1
bMatrix:          input matrix 2 array on device
bRows:            rows of input matrix 2
bCols:            columns of input matrix 2
cMatrix:          output matrix array on device
sigm:             Whether to do sigmoid() on the result 
*/


void downsample_bbox(float* bboxDevice, int length, float heightRatio, float widthRatio, cudaStream_t stream);


void crop_mask(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* bboxesDevice, cudaStream_t stream);
/*
    set value 0 where masks out of bboxes
masksDevice:      mask array on device, shape(n, 160 x 160)
maskNum:          n, number of masks
maskHeight:       height of each mask
maskWidth:        width of each mask
bboxesDevice:     bbox array on device, shape(n, 4), 4 : x1, y1, x2, y2
*/


void cut_mask(
    float* masksDevice, int maskNum, int maskHeight, int maskWidth,
    float* cutMasksDevice, int cutMaskTop, int cutMaskLeft, int cutMaskH, int cutMaskW, cudaStream_t stream
);


void resize(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* dstMasksDevice, int dstMaskH, int dstMaskW, cudaStream_t stream);


__inline__ void scale_bbox(cv::Mat& img, float bbox[4]){
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (kInputH - r * img.rows) / 2;
    float pad_w = (kInputW - r * img.cols) / 2;

    bbox[0] = (bbox[0] - pad_w) / r;
    bbox[1] = (bbox[1] - pad_h) / r;
    bbox[2] = (bbox[2] - pad_w) / r;
    bbox[3] = (bbox[3] - pad_h) / r;
}


#endif  // POSTPROCESS_H
