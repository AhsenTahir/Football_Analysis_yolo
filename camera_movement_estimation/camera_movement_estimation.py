import cv2
import sys
import numpy as np
import gc
sys.path.append('../')
from utils import measure_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.min_distance = 5
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )
        # lucas kanade params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def add_adjust_position_to_track(self, track, camera_movement_per_frame):
        for object, object_track in track.items():
            for frame_num, track in enumerate(object_track):
                camera_movement = camera_movement_per_frame[frame_num]
                for track_id, track_info in track.items():
                    position = track_info.get("position")  # Use get to avoid KeyError if "position" is missing
                    if position is not None:
                        adjusted_position = (position[0] + camera_movement[0], position[1] + camera_movement[1])
                        track_info["adjusted_position"] = adjusted_position
                    else:
                        print(f"Warning: 'position' not found in track_info for track_id {track_id}")
        return track

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        camrea_movement = [[0, 0] for i in range(len(frames))]
        previous_gray_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_gray_frame_features = cv2.goodFeaturesToTrack(previous_gray_frame, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(previous_gray_frame, frame_gray, previous_gray_frame_features, None, **self.lk_params)

            max_distance = 0
            camere_movement_x, camrea_movement_y = 0, 0
            for i, (new, old) in enumerate(zip(new_features, previous_gray_frame_features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()
                distance = measure_distance(new_feature_point, old_feature_point)
                if distance > max_distance:
                    max_distance = distance
                    camere_movement_x, camrea_movement_y = new_feature_point - old_feature_point
            if max_distance > self.min_distance:
                camrea_movement[frame_num] = [camere_movement_x, camrea_movement_y]
                previous_gray_frame_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            previous_gray_frame = frame_gray.copy()

        return camrea_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        try:
            for frame_num, frame in enumerate(frames):
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # In-place modification

                x_movement, y_movement = camera_movement_per_frame[frame_num]
                frame = cv2.putText(frame, f"X Movement: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                frame = cv2.putText(frame, f"Y Movement: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                output_frames.append(frame)

                if frame_num % 10 == 0:
                    gc.collect()  # Collect garbage to free up memory
        finally:
            gc.collect()

        return output_frames
