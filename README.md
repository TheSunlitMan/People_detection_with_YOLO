# People Detection in Video using YOLOv8
This project detects people in a video using the pre-trained YOLOv8n model, draws bounding boxes with confidence scores, and saves the annotated output video.

## Project Structure

```
.
├── main.py                   # Main script
├── requirements.txt          # Dependencies
├── README.md
├── data/
│   ├── input/                # Place input videos here (e.g., crowd.mp4)
│   └── output/               # Processed videos will be saved here
└── models/                   # Model weights will be downloaded here automatically
```

## Requirements

- Python 3.9+
- `pip` (or `conda`)

Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: The first run will automatically download `yolov8n.pt` into the `models/` directory.

## Usage

Place your input video in `data/input/` (default: `crowd.mp4`), then run:

```bash
python main.py
```

Or specify custom paths:

```bash
python detect_people.py \
  --input data/input/my_video.mp4 \
  --output data/output/result.mp4 \
  --model models/yolov8n.pt
```

### Arguments
| Argument | Default | Description |
|--------|--------|-------------|
| `--input` | `data/input/crowd.mp4` | Path to input video |
| `--output` | `data/output/output.mp4` | Path to save processed video |
| `--model` | `models/yolov8n.pt` | Path to YOLO model (will be downloaded if missing) |

## Dependencies

- [`ultralytics>=8.4.0`](https://github.com/ultralytics/ultralytics) — YOLOv8 implementation  
- [`opencv-python>=4.12.0`](https://opencv.org/) — Video I/O and drawing  
- [`torch>=2.9.0`](https://pytorch.org/) — Deep learning backend  

See full list in [`requirements.txt`](requirements.txt).

## Quality Notes & Improvement Ideas

- **Strengths**:  
  - Real-time capable on modern hardware.  
  - High accuracy for upright, full-body people in well-lit scenes.

- **Limitations observed**:  
  - Small or distant people may be missed.  
  - Overlapping individuals can cause partial detection or duplication.  
  - No tracking → flickering IDs between frames.

- **Possible improvements**:  
  1. Use a larger model (`yolov8m` or `yolov8l`) for higher accuracy.  
  2. Add object tracking (e.g., ByteTrack via `model.track()`) to stabilize detections.  
  3. Fine-tune the model on domain-specific data (e.g., crowded metro scenes).  
  4. Optimize for edge cases: low light, occlusions, unusual poses.

## 📎 License

This project is for educational and evaluation purposes only.
