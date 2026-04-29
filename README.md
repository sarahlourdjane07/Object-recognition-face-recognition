# Object & Face Recognition App

A desktop application built with Python and Tkinter that performs real-time **object detection** (YOLOv8) and **face recognition** via webcam or screen share.

## Features

- **Object Detection** — detects objects in real time using YOLOv8, draws bounding boxes with class labels and confidence scores
- **Face Recognition** — identifies known people from the `People/` folder with confidence percentage
- **Screen Share Mode** — captures the screen instead of a webcam and runs face recognition on it
- **Live FPS counter** — displayed on the video feed

## Project Structure

```
├── main.py               # Entry point, Tkinter UI
├── Camera_Module.py      # Webcam capture and frame loop
├── yolo_detector.py      # YOLOv8 object detection wrapper
├── face_recogmodule.py   # Face recognition + screen capture
├── yolov8n.pt            # YOLOv8 nano weights
├── People/               # Folder with photos of known people
│   └── Name.jpg          # One photo per person (filename = name)
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sarahlourdjane07/Object-recognition-face-recognition.git
cd Object-recognition-face-recognition
```

### 2. Install dependencies

> Requires Python 3.10

```bash
pip install -r requirements.txt
```

> **Windows only:** If `face-recognition` / `dlib` fails to install, use the bundled wheel:
> ```bash
> pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
> pip install face-recognition
> ```

### 3. Add known faces

Create a `People/` folder and put photos of people you want to recognize:

```
People/
├── Alice.jpg
├── Bob.png
└── ...
```

One face per photo. The filename (without extension) is used as the person's name.

## Usage

```bash
python main.py
```

| Button | Action |
|---|---|
| Start Camera | Opens webcam with object detection |
| Stop Camera | Stops webcam |
| Screenshare | Captures screen and runs face recognition |
| Stop Screenshare | Stops screen capture |

## Requirements

- Python 3.10
- Webcam (for camera mode)
- Windows recommended for screen capture (dxcam backend); macOS/Linux use mss or pyautogui
