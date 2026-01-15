"""
People Detection in Video using YOLOv8

This script loads a pre-trained YOLOv8 model, processes an input video,
detects people (class 0 in COCO), draws bounding boxes with confidence scores,
and saves the annotated video to the output path.
"""
import os
import argparse
import torch
import cv2
from ultralytics import YOLO


def parse_arguments():
    """
    Parse command-line arguments for input and output video paths.

    Returns:
        argparse.Namespace: Parsed arguments with 'input' and 'output' attributes.
    """
    parser = argparse.ArgumentParser(
        description="Detect people in a video and draw bounding boxes with confidence scores."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/crowd.mp4",
        help="Path to the input video file (default: data/input/crowd.mp4)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/output.mp4",
        help="Path to save the processed video (default: data/output/output.mp4)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path to the YOLO model weights (default: yolov8n.pt)."
    )
    return parser.parse_args()


def load_model(model_path: str):
    """
    Load a pre-trained YOLOv8 model from the given path.
    If the file does not exist, download it to that location first.

    Args:
        model_path (str): Path to the model weights (.pt file).

    Returns:
        ultralytics.YOLO: Loaded YOLO model ready for inference.

    Raises:
        RuntimeError: If downloading or loading fails.
    """
    # Создаём директорию, если её нет
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Если файл не существует — скачиваем
    if not os.path.isfile(model_path):
        print(f"Model not found at '{model_path}'. Downloading...")
        try:
            # Официальный URL для yolov8n.pt (актуален на 2025–2026 гг.)
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
            torch.hub.download_url_to_file(url, model_path, progress=True)
            print(f"✅ Model successfully downloaded to '{model_path}'")
        except Exception as download_error:
            raise RuntimeError(f"Failed to download model from {url}: {download_error}")

    # Загружаем модель
    try:
        model = YOLO(model_path)
        return model
    except Exception as load_error:
        raise RuntimeError(f"Failed to load model from '{model_path}': {load_error}")


def process_video(input_path: str, output_path: str, model):
    """
    Process video frame-by-frame: detect people and annotate bounding boxes.

    Only class 0 ('person') from COCO is used.
    Bounding boxes are green with class label and confidence score above them.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save annotated output video.
        model (ultralytics.YOLO): Loaded YOLO model.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure FPS is valid
    if fps <= 0:
        fps = 30.0  # fallback

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference: only class 0 (person)
        results = model(frame, classes=[0], verbose=False)

        # Annotate each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"person: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background (optional but improves readability)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point of the application."""
    args = parse_arguments()

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    print(f"Processing video: {args.input} → {args.output}")
    process_video(args.input, args.output, model)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()