# 🎥 Multi-Video Single-Object Real-Time Tracker

This project tracks a single moving object across **multiple videos** using OpenCV object trackers 
(CSRT, KCF, MOSSE), with optional **auto-initialization** via motion detection and **Kalman filter smoothing**.

## 🎥 Demo
![Demo](demo-small.gif)

---

## 🚀 Features
- Batch-process multiple videos (input folder or specific files).
- Track with **CSRT**, **KCF**, or **MOSSE** trackers.
- **Manual ROI selection** (select bounding box on first frame).
- **Auto-init ROI** using motion detection on first N frames.
- **Kalman filter smoothing** of object trajectory.
- Outputs:
  - Annotated `.mp4` video with bounding boxes + trajectories.
  - `.csv` file with per-frame tracking info (`frame, timestamp, bbox, centroid`).

---


## 📂 Project Structure
```
multi-video-tracker/
│── .venv/ # Python virtual environment
│── videos/ # input videos
│── runs/ # results (auto-generated)
│── multi_video_tracker.py 
│── requirements.txt # Python dependencies
│── README.md # documentation
```

## 📂 Project Structure
```
object-tracking-project/
│── videos/                  # raw input videos
│── tracking_data/           # 10 CSV file
│                  
│── main.ipynb               # where all csv file turn into one dataset for applying ml model 
```



## 🛠️ Setup

1. **Clone project & create venv**
   ```bash
   git clone <repo-url> multi-video-tracker
   cd multi-video-tracker
   python -m venv .venv

2. **Activate virtual environment**
   ```bash
   .venv\Scripts\Activate

3. **Install requirements**
    ```bash
     pip install -r requirements.txt
 
**Run the Tracker**

Manual ROI Selection/Automatic ROI
  ```bash
  python multi_video_tracker.py --videos ./videos --output ./runs/track --tracker csrt --display
  


  
  python multi_video_tracker.py --videos ./videos --output ./runs/track --auto-init --auto-frames 60


📊 **Output**

For each input video:

video_tracked.mp4 → annotated with bounding boxes + trajectories

video_track.csv → per-frame tracking info:

   
   frame	time_s	x	y	w	h	cx	cy
   0	0.0000	123.0	200.0	50.0	80.0	148.0	240.0

⚡ **Notes**

opencv-contrib-python is required for CSRT/MOSSE trackers.
If tracking fails, you’ll see "Tracking lost" message on video.
