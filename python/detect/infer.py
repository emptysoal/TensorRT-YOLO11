# -*- coding:utf-8 -*-

"""
    onnx 模型转 tensorrt 模型，并使用 tensorrt python api 推理
"""

import os

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

from config import *
from preprocess import preprocess
from postprocess import postprocess
import calibrator


class YoloDetector:
    def __init__(
            self,
            trt_plan=trt_file,
            gpu_id=kGpuId,
            num_classes=kNumClass,
            nms_thresh=kNmsThresh,
            conf_thresh=kConfThresh
    ):
        self.trt_file = trt_plan
        self.logger = trt.Logger(trt.Logger.ERROR)
        cudart.cudaSetDevice(gpu_id)

        self.nums_classes = num_classes
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

        self.engine = self.get_engine()

        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(0, [1, 3, kInputH, kInputW])

        n_io = self.engine.num_bindings
        self.buffer_h = []
        for i in range(n_io):
            self.buffer_h.append(
                np.empty(self.context.get_binding_shape(i), dtype=trt.nptype(self.engine.get_binding_dtype(i))))
        self.buffer_d = []
        for i in range(n_io):
            self.buffer_d.append(cudart.cudaMalloc(self.buffer_h[i].nbytes)[1])

    def release(self):
        for b in self.buffer_d:
            cudart.cudaFree(b)

    def get_engine(self):
        if os.path.exists(self.trt_file):
            with open(self.trt_file, "rb") as f:  # read .plan file if exists
                engine_string = f.read()
            if engine_string is None:
                print("Failed getting serialized engine!")
                return
            print("Succeeded getting serialized engine!")
        else:
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # set workspace for TensorRT
            if use_fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if use_int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator.MyCalibrator(calibration_data_dir, n_calibration,
                                                                 (8, 3, kInputW, kInputW), cache_file)

            parser = trt.OnnxParser(network, self.logger)
            if not os.path.exists(onnx_file):
                print("Failed finding ONNX file!")
                return
            print("Succeeded finding ONNX file!")
            with open(onnx_file, "rb") as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return
                print("Succeeded parsing .onnx file!")

            input_tensor = network.get_input(0)
            profile.set_shape(input_tensor.name, [1, 3, kInputH, kInputW], [1, 3, kInputH, kInputW],
                              [1, 3, kInputH, kInputW])
            config.add_optimization_profile(profile)

            engine_string = builder.build_serialized_network(network, config)
            if engine_string is None:
                print("Failed building engine!")
                return
            print("Succeeded building engine!")
            with open(self.trt_file, "wb") as f:
                f.write(engine_string)

        engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)

        return engine

    def inference_one(self, data_input, context, buffer_h, buffer_d):
        """
            使用tensorrt runtime 做一次推理
        """
        buffer_h[0] = np.ascontiguousarray(data_input)
        cudart.cudaMemcpy(buffer_d[0], buffer_h[0].ctypes.data, buffer_h[0].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(buffer_d)  # inference

        cudart.cudaMemcpy(buffer_h[1].ctypes.data, buffer_d[1], buffer_h[1].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        outs = buffer_h[-1].reshape((4 + self.nums_classes, -1))

        return outs

    def inference(self, image):
        input_data = preprocess(image, kInputH, kInputW)  # image preprocess
        input_data = np.expand_dims(input_data, axis=0)  # add batch size dimension

        output = self.inference_one(input_data, self.context, self.buffer_h, self.buffer_d)

        detect_res = postprocess(image, output, self.conf_thresh, self.nms_thresh, kInputW, kInputW)

        return detect_res

    @staticmethod
    def draw_image(detected_res, image, line_color=(255, 0, 255), label_color=(255, 255, 255), line_thickness=2):
        for x1, y1, x2, y2, conf, class_id in detected_res:
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(image, c1, c2, line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

            label = f"{class_name_list[int(class_id)]} {conf:.2f}"
            # label = class_name_list[int(class_id)]
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=line_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, line_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, label_color, thickness=line_thickness,
                        lineType=cv2.LINE_AA)
