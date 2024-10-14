# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

from preprocess import preprocess


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_path, n_calibration, input_shape, cache_file):
        super(MyCalibrator, self).__init__()
        self.image_list = []
        self.n_calibration = n_calibration
        self.shape = input_shape  # (N,C,H,W)
        self.buffer_size = trt.volume(input_shape) * trt.float32.itemsize
        self.cache_file = cache_file
        _, self.d_in = cudart.cudaMalloc(self.buffer_size)
        self.one_batch = self.batch_generator()

        for per_image_name in os.listdir(calibration_data_path):
            per_image_path = os.path.join(calibration_data_path, per_image_name)
            self.image_list.append(per_image_path)

        print(int(self.d_in))

    def __del__(self):
        cudart.cudaFree(self.d_in)

    @staticmethod
    def image_preprocess(image_path: str, input_size: tuple):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        data = preprocess(img, input_size[0], input_size[1])

        return data

    def batch_generator(self):
        for i in range(self.n_calibration):
            print("> calibration %d" % i)
            sub_image_list = np.random.choice(self.image_list, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.load_image_list(sub_image_list))

    def load_image_list(self, image_list):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            res[i] = self.image_preprocess(image_list[i], tuple(self.shape[2:]))
        return res

    def get_batch_size(self):  # necessary API
        return self.shape[0]

    def get_batch(self, name_list=None, input_node_name=None):  # necessary API
        try:
            data = next(self.one_batch)
            cudart.cudaMemcpy(self.d_in, data.ctypes.data, self.buffer_size,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.d_in)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # necessary API
        if os.path.exists(self.cache_file):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Finding no int8 cache!")
            return

    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return


if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator("../../../../flower_classify_dataset/val/", 5, (1, 3, 640, 640), "./int8.cache")
    m.get_batch("FakeNameList")
