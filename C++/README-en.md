# YOLO11  TensorRT  ByteTrack

![street](../assets/result.gif)



## RUN

1. Go to the `detect` directory and convert the model according to `README`
2. Based on the `detect` environment, install the following libraries:

```bash
apt install libeigen3-dev
```

3. Switch to current directory, start track：

```bash
mkdir build
cd build
cmake ..
make
./main ../assets/street.mp4  # your own video path
```

- The result is saved in the current directory as `result.mp4`;
- If you want the tracked video to play in real time, you can uncomment lines 93-95 of `main.cpp` 

