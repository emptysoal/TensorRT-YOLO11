#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"
#include "draw.h"

using namespace nvinfer1;


YoloDetector::YoloDetector(const std::string trtFile): trtFile_(trtFile)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(kGpuId);

    CHECK(cudaStreamCreate(&stream));

    // load engine
    get_engine();

    context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, kInputH, kInputW}});

    // get engine output info
    protoOutDims = context->getBindingDimensions(1);  // proto [1 32 160 160]
    int protoOutputSize = 1;  // 32 * 160 * 160
    for (int i = 0; i < protoOutDims.nbDims; i++){
        protoOutputSize *= protoOutDims.d[i];
    }

    Dims32 outDims = context->getBindingDimensions(2);  // [1 116 8400], 116 = 4 + 80 + 32 = bbox + class + mask coefficients
    OUTPUT_CANDIDATES = outDims.d[2];  // 8400
    int outputSize = 1;  // 116 * 8400
    for (int i = 0; i < outDims.nbDims; i++){
        outputSize *= outDims.d[i];
    }

    // prepare output data space on host
    outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    // prepare input and output space on device
    vBufferD.resize(3, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[1], protoOutputSize * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[2], outputSize * sizeof(float)));

    CHECK(cudaMalloc(&transposeDevide, outputSize * sizeof(float)));
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
}

void YoloDetector::get_engine(){
    if (access(trtFile_.c_str(), F_OK) == 0){
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return; }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode){
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity))){
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {1, 3, kInputH, kInputW}});
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr){
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile_, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;

        delete engineString;
        delete parser;
        delete config;
        delete network;
        delete builder;
    }
}

YoloDetector::~YoloDetector(){
    cudaStreamDestroy(stream);

    for (int i = 0; i < 3; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    CHECK(cudaFree(transposeDevide));
    CHECK(cudaFree(decodeDevice));

    delete [] outputData;

    delete context;
    delete engine;
    delete runtime;
}

std::vector<Detection> YoloDetector::inference(cv::Mat& img){
    if (img.empty()) return {};

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);

    // tensorrt inference
    context->enqueueV2(vBufferD.data(), stream, nullptr);

    // transpose [116 8400] convert to [8400 116]
    transpose((float*)vBufferD[2], transposeDevide, OUTPUT_CANDIDATES, 4 + kNumClass + 32, stream);
    // convert [8400 116] to [39001, ], 39001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 32masks)
    decode(transposeDevide, decodeDevice, OUTPUT_CANDIDATES, kNumClass, 32, kConfThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    // cuda nms
    nms(decodeDevice, kNmsThresh, kMaxNumOutputBbox, kNumBoxElement, stream);
    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // cudaStreamSynchronize(stream);

    std::vector<Detection> vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox);
    for (int i = 0; i < count; i++){
        int pos = 1 + i * kNumBoxElement;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1){
            Detection det;
            memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
            det.conf = outputData[pos + 4];
            det.classId = (int)outputData[pos + 5];
            memcpy(det.mask, &outputData[pos + 7], 32 * sizeof(float));
            vDetections.push_back(det);
        }
    }

    process_mask((float*)vBufferD[1], protoOutDims, vDetections, kInputH, kInputW, img, stream);
    cudaStreamSynchronize(stream);

    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
    }

    return vDetections;
}

void YoloDetector::process_mask(
    float* protoDevice, Dims32 protoOutDims, std::vector<Detection>& vDetections, 
    int kInputH, int kInputW, cv::Mat& img, cudaStream_t stream
){
    int protoC = protoOutDims.d[1];  // default 32
    int protoH = protoOutDims.d[2];  // default 160
    int protoW = protoOutDims.d[3];  // default 160

    int n = vDetections.size();  // number of bboxes
    if (n == 0) return;

    // prepare n x 32 length mask coef space on device
    float* maskCoefDevice = nullptr;
    CHECK(cudaMalloc(&maskCoefDevice, n * protoC * sizeof(float)));
    // prepare n x 160 x 160 mask space on device
    float* maskDevice = nullptr;
    CHECK(cudaMalloc(&maskDevice, n * protoH * protoW * sizeof(float)));

    float* bboxDevice = nullptr;  // x1,y1,x2,y2,x1,y1,x2,y2,...x1,y1,x2,y2
    CHECK(cudaMalloc(&bboxDevice, n * 4 * sizeof(float)));

    for (size_t i = 0; i < n; i++){
        CHECK(cudaMemcpyAsync(&maskCoefDevice[i * protoC], vDetections[i].mask, protoC * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(&bboxDevice[i * 4], vDetections[i].bbox, 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    // mask = sigmoid(mask coef x proto)
    matrix_multiply(maskCoefDevice, n, protoC, protoDevice, protoC, protoH * protoW, maskDevice, stream, true);

    // down sample bbox from 640x640 to 160x160
    float heightRatio = (float)protoH / (float)kInputH;  // 160 / 640 = 0.25
    float widthRatio = (float)protoW / (float)kInputW;  // 160 / 640 = 0.25
    downsample_bbox(bboxDevice, n * 4, heightRatio, widthRatio, stream);

    // set 0 where mask out of bbox
    crop_mask(maskDevice, n, protoH, protoW, bboxDevice, stream);

    // scale mask from 160x160 to original resolution
    // 1. cut mask
    float r_w = protoW / (img.cols * 1.0);
    float r_h = protoH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (protoH - r * img.rows) / 2;
    float pad_w = (protoW - r * img.cols) / 2;
    int cutMaskLeft = (int)pad_w;
    int cutMaskTop = (int)pad_h;
    int cutMaskRight = (int)(protoW - pad_w);
    int cutMaskBottom = (int)(protoH - pad_h);
    int cutMaskWidth = cutMaskRight - cutMaskLeft;
    int cutMaskHeight = cutMaskBottom - cutMaskTop;
    float* cutMaskDevice = nullptr;
    CHECK(cudaMalloc(&cutMaskDevice, n * cutMaskHeight * cutMaskWidth * sizeof(float)));
    cut_mask(maskDevice, n, protoH, protoW, cutMaskDevice, cutMaskTop, cutMaskLeft, cutMaskHeight, cutMaskWidth, stream);

    // 2. bilinear resize mask
    float* scaledMaskDevice = nullptr;
    CHECK(cudaMalloc(&scaledMaskDevice, n * img.rows * img.cols * sizeof(float)));
    resize(cutMaskDevice, n, cutMaskHeight, cutMaskWidth, scaledMaskDevice, img.rows, img.cols, stream);

    for (size_t i = 0; i < n; i++){
        vDetections[i].maskMatrix.resize(img.rows * img.cols);
        CHECK(cudaMemcpyAsync(vDetections[i].maskMatrix.data(), &scaledMaskDevice[i * img.rows * img.cols], img.rows * img.cols * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    CHECK(cudaFree(maskCoefDevice));
    CHECK(cudaFree(maskDevice));
    CHECK(cudaFree(bboxDevice));
    CHECK(cudaFree(cutMaskDevice));
    CHECK(cudaFree(scaledMaskDevice));
}


void YoloDetector::draw_image(cv::Mat& img, std::vector<Detection>& inferResult, bool drawBbox){
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++){
        // draw bboxes
        if (drawBbox){
            cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
            cv::Rect r(
                round(inferResult[i].bbox[0]),
                round(inferResult[i].bbox[1]),
                round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
                round(inferResult[i].bbox[3] - inferResult[i].bbox[1])
            );
            cv::rectangle(img, r, bboxColor, 2);

            std::string className = vClassNames[(int)inferResult[i].classId];
            std::string labelStr = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

            cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
            cv::Point topLeft(r.x, r.y - textSize.height - 3);
            cv::Point bottomRight(r.x + textSize.width, r.y);
            cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
            cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }

        draw_mask(img, inferResult[i].maskMatrix.data());
    }
}
