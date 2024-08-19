import cv2
#it will return the frames of the video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()#ret is a boolean value that returns true if the frame is available
        if not ret:
            break
        frames.append(frame)

    return frames    

#it will make a video from the frames
def save_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release() 