#ifndef TYPES_H
#define TYPES_H

#include <string>
#include "config.h"


struct Detection
{
    // x1, y1, x2, y2
    float bbox[4];  // bbox both before and after scale
    float conf;
    int classId;
    float kpts[kNumKpt * kKptDims];  // key points: 17 * 3 = 51, before scale to original image
    std::vector<std::vector<float>> vKpts;  // key points after scale: {{x, y, visible}, {x, y, visible}, {x, y, visible}, ...}
};


#endif  // TYPES_H
