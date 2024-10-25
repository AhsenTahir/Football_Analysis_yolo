import cv2

#it will return the frames of the video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames    

#it will make a video from the frames
def save_video(frames, output_path):
    if not frames:
        raise ValueError("No frames to save")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
