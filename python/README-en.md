# YOLO11  TensorRT  ByteTrack

![street](../assets/result.gif)



## RUN

1. Go to the `detect` directory and convert the model according to `README`
2. Based on the `detect` environment, install the following libraries:

```bash
pip install lap
pip install cython_bbox
```

3. start track：

```bash
python track.py --video ./videos/street.mp4
```

- The result is saved in the current directory as `result.mp4`;
- If you want the tracked video to play in real time, you can uncomment lines 108-111 of `track.py` 

