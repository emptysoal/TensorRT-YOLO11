# YOLO11 关键点检测模型转 TensorRT

![_10020](output/_10020.jpg)

## 导出ONNX模型

1. 安装 `YOLO11`

```bash
pip install ultralytics
```

- 建议同时从 `GitHub` 上 clone 或下载一份 `YOLO11` 源码到本地；
- 根据 `GitHub` 上 `YOLO11` 的 `README` 中的链接下载检测模型，如：`yolo11s-pose.pt`；
- 在本地 `YOLO11`一级 `ultralytics` 目录下，新建 `weights` 目录，并且放入下载的`yolo11s-pose.pt`模型

2. 安装onnx相关库

```bash
pip install onnx==1.14.0
pip install onnxslim==0.1.34
```

3. 导出onnx模型

- 可以在一级 `ultralytics` 目录下，新建 `export_onnx.py` 文件
- 向文件中写入如下内容：

```python
from ultralytics import YOLO

model = YOLO("./weights/yolo11s-pose.pt")
path = model.export(format="onnx", simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
```

- 运行 `python export_onnx.py` 后，会在 `weights` 目录下生成 `yolo11s-pose.onnx`

## 转 TensorRT

1. 切换到当前项目目录下；
2. 如果是自己数据集上训练得到的模型，记得更改 `include/config.h` 中的相关配置；
3. 确认 `CMakeLists.txt` 文件中 `cuda` 和 `tensorrt` 库的路径，与自己环境要对应，一般情况下是不需修改的；
4. 新建 `onnx_model`目录，并将已导出的 `onnx` 模型拷贝到 `onnx_model` 目录下
5. 依次执行：

```bash
mkdir build
cd build
cmake ..
make
./main ../images  # 不存在trt模型时，会先构建trt模型然后推理；存在trt模型时，直接加载trt模型然后推理
```

之后转换后的模型，以及首次 TensorRT 的推理结果都会保存到当前目录下

