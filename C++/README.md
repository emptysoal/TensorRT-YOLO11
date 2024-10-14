# YOLO11  TensorRT  ByteTrack

![street](../assets/result.gif)



## 运行

1. 进入 `detect` 目录，按照其中 `README` 完成模型转换；
2. 在 `detect` 环境的基础上，安装 `Eigen` 库：

```bash
apt install libeigen3-dev
```

3. 切换到当前目录，启动跟踪：

```bash
mkdir build
cd build
cmake ..
make
./main ../assets/street.mp4  # 传入自己视频的路径
```

- 结果会在当前目录保存为 `result.mp4`;
- 如果想要跟踪的视频实时播放，可解开`main.cpp`第 93 ~ 95 行的注释