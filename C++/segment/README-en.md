# TensorRT deploy YOLO11 segment

![_10020](output/_10014.jpeg)

## Get onnx

```bash
pip install ultralytics
pip install onnx==1.14.0
pip install onnxslim==0.1.34
```

write export_onnx.py, as follow:

```python
from ultralytics import YOLO

model = YOLO("./weights/yolo11s-seg.pt")
path = model.export(format="onnx", simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
```

run export_onnx.py

```bash
python export_onnx.py
```

`yolo11s-seg.onnx` will be generated

## To TensorRT

1. Switch to the current project directory;
2. If the model is trained on your own dataset, remember to check `include/config.h`;
3. Confirm `cuda` and `tensorrt` path in  `CMakeLists.txt`
4. Create `onnx_model` directory and put the exported `onnx` model in
5. Run as follow:

```bash
mkdir build
cd build
cmake ..
make
./main ../images
```