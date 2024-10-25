import sys
sys.path.append('..')
from utils import measure_distance,getFootPosition
import cv2

class SpeedAnddistanceEstimator():
    def __init__(self):
        self.frameWindow = 30  # Increased from 5 to 30
        self.frameRate = 24
        self.pixelsPerMeter = 30  # This needs to be calibrated for your specific video
        self.speedSmoothingWindow = 5  # For moving average

    def addSpeedAndDistance(self, tracks):
        total_distance = {}
        speed_history = {}

        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referees':
                continue

            numberOfFrames = len(object_tracks)

            for frame_num in range(0, numberOfFrames, self.frameWindow):
                last_frame = min(frame_num + self.frameWindow, numberOfFrames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    # Convert pixel distance to meters
                    distance_covered = measure_distance(start_position, end_position) / self.pixelsPerMeter
                    time_elapsed = (last_frame - frame_num) / self.frameRate

                    if time_elapsed <= 0:
                        continue

                    speed = distance_covered / time_elapsed  # Speed in m/s
                    speed_in_kmph = speed * 3.6  # Convert to km/h

                    # Initialize speed history if not already done
                    if track_id not in speed_history:
                        speed_history[track_id] = []

                    # Add current speed to history
                    speed_history[track_id].append(speed_in_kmph)

                    # Keep only the last 'speedSmoothingWindow' speeds
                    speed_history[track_id] = speed_history[track_id][-self.speedSmoothingWindow:]

                    # Calculate average speed
                    avg_speed = sum(speed_history[track_id]) / len(speed_history[track_id])

                    # Initialize distance tracking if not already done
                    if track_id not in total_distance:
                        total_distance[track_id] = 0

                    # Accumulate total distance
                    total_distance[track_id] += distance_covered

                    # Update speed and distance for each frame in the current window
                    for frame_num_batch in range(frame_num, last_frame + 1):
                        if track_id not in object_tracks[frame_num_batch]:
                            continue
                        object_tracks[frame_num_batch][track_id]['speed'] = avg_speed
                        object_tracks[frame_num_batch][track_id]['distance'] = total_distance[track_id]

        return tracks

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for track_id, track in object_tracks[frame_num].items():
                    if "speed" in track:
                        speed = track.get('speed', None)
                        distance = track.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        bbox = track['bbox']
                        position = getFootPosition(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
