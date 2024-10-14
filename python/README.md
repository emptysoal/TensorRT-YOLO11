# YOLO11  TensorRT  ByteTrack

![street](../assets/result.gif)



## 运行

1. 进入 `detect` 目录，按照其中 `README` 转换完模型；
2. 在 `detect` 环境的基础上，安装以下库：

```bash
pip install lap
pip install cython_bbox
```

3. 启动跟踪：

```bash
python track.py --video ./videos/street.mp4
```

- 结果会在当前目录保存为 `result.mp4`;
- 如果想要跟踪的视频实时播放，可解开`track.py`第 108 ~ 111 行的注释