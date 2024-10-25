# VisionPlay: Next-Gen Sports Analytics using Deep Learning
VisionPlay is an advanced sports analytics system leveraging deep learning and computer vision to analyze soccer match footage in real-time. By tracking players, referees, and the ball, VisionPlay provides actionable insights into player performance, team strategies, and game dynamics.

## Project Overview
VisionPlay utilizes state-of-the-art algorithms to analyze soccer videos, offering comprehensive features for coaches, analysts, and fans:

- **Multi-Object Detection and Tracking**
- **Player Identification and Team Assignment**
- **Ball Possession Analysis**
- **Performance Metrics Calculation**
- **Camera Movement Compensation**
- **Perspective Transformation**
- **Visual Annotations**

---

## Key Components and Algorithms

### 1. Multi-Object Detection and Tracking
- **Object Detection**: Uses a fine-tuned YOLO (You Only Look Once) model to detect players, goalkeepers, referees, and the ball in real-time.
- **Object Tracking**: Employs ByteTrack for robust tracking across frames, ensuring accurate and stable tracking.

### 2. Player Identification and Team Assignment
- **Team Assignment**: Uses K-means clustering to assign players to teams based on jersey color.
- **Player Tracking**: Uniquely identifies and tracks individual players throughout the match.

### 3. Ball Possession Analysis
- **Ball-Player Association**: Identifies the player in possession of the ball for each frame.
- **Team Possession Statistics**: Calculates and displays real-time team ball control.

### 4. Performance Metrics Calculation
- **Speed Calculation**: Computes player speed based on movement between frames.
- **Distance Tracking**: Tracks total distance covered by each player.

### 5. Camera Movement Compensation
- **Optical Flow**: Estimates camera movement and applies compensation for accurate player positions.

### 6. Perspective Transformation
- **Coordinate Mapping**: Maps pixel coordinates to real field positions for spatial analysis.

### 7. Visual Annotations
- **Bounding Boxes**: Highlights players, referees, and the ball.
- **Performance Metrics Display**: Shows speed, distance, and team possession in real-time.

---
#Training
The YOLO model used for object detection was fine-tuned on a custom dataset containing labeled players, goalkeepers, referees, and the ball. Details of the training process can be found in the notebooks/Training_YOLO.ipynb notebook.
## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VisionPlay.git
cd VisionPlay
```
## Install Dependencies
Install the necessary Python libraries by running:
```bash
pip install -r requirements.txt
```
## Running VisionPlay
To start analyzing a video with VisionPlay:

Place your video files in the videos/ folder.
Run the main script to analyze the footage.
