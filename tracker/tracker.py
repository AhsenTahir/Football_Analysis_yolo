from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import os
import sys 
import cv2
import pandas as pd
sys.path.append('.../')
from utils import get_center_of_the_box, get_width_of_the_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def ball_interpolation(self, ball_tracks):
        # ball_tracks is a list of dictionaries where each dictionary has only one key-value pair
        # The key is the track ID and the value is a dictionary containing the bbox
        # Example:
        # [
        #     {1: {"bbox": [100, 200, 150, 250]}},
        #     {1: {"bbox": [110, 210, 160, 260]}},
        #     {1: {"bbox": [120, 220, 170, 270]}},
        #     {1: {"bbox": [130, 230, 180, 280]}},
        #     {1: {"bbox": [140, 240, 190, 290]}},
        #     {1: {"bbox": [150, 250, 200, 300]}},
        #     {1: {"bbox": [160, 260, 210, 310]}},
        #     {1: {"bbox": [170, 270, 220, 320]}},
        #     {1: {"bbox": [180, 280, 230, 330]}},
        #     {1: {"bbox": [190, 290, 240, 340]}},
        # ]
        # We need to interpolate the missing values in the bbox
        ball_positions=[x.get(1,{}).get("bbox",[]) for x in ball_tracks]
        df_ball_positions=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions=df_ball_positions.bfill()
        ball_positions=[{1:{"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)#returns a list of dictionaries having bbox,confidence and class name
            # self.model.to('cuda')
            detections += detection_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        else:
            detections = self.detect_frames(frames)
            
            
            tracks = {
                "player": [{} for _ in range(len(frames))],
                "referee": [{} for _ in range(len(frames))],
                "ball": [{} for _ in range(len(frames))]
            }

            for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)
                print("Detection Supervision Attributes:", dir(detection_supervision))
                

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":#this is done to make goalkeeper and player the same thing for tracking
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

                detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
                # detections_with_tracks = [
                #     ([100, 150, 200, 250], 0.9, some_value, 0, track_id_1),  # Player tracked with track_id_1
                #     ([300, 400, 350, 450], 0.8, some_value, 1, track_id_2)   # Referee tracked with track_id_2
                # ]


                for frame_detection in detections_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    if cls_id == cls_names_inv["player"]:
                        tracks["player"][frame_num][track_id] = {"bbox": bbox}
                    if cls_id == cls_names_inv["referee"]:
                        tracks["referee"][frame_num][track_id] = {"bbox": bbox}

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    if cls_id == cls_names_inv["ball"]:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
            
            if stub_path is not None:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)
            return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # bbox is in the format [top left x, top left y, bottom right x, bottom right y]
        y2 = int(bbox[3])
        x_center, _ = get_center_of_the_box(bbox)
        width = get_width_of_the_box(bbox)
        height = int(width * 0.35)  # Height is 35% of the width

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(height)),
                    # angle=-45,  # Rotation angle of the ellipse
                    angle=-5,
                    startAngle=0,  # Start angle of the arc
                    endAngle=235,  # End angle of the arc
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)  # Antialiased line
        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center-rectangle_width//2
        x2_rect=x_center+rectangle_width//2
        y1_rect=(y2-rectangle_height//2)+15
        y2_rect=(y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,
                           (int(x1_rect),int(y1_rect)),
                           (int(x2_rect),int(y2_rect)),
                           color,
                           cv2.FILLED)
            x1_text=x1_rect+12
            if track_id is not None and track_id > 99:
                x1_text-=10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2)
        return frame
    def draw_triangle(self, frame, bbox, color, track_id=None):
        y= int(bbox[1])
        x, _ = get_center_of_the_box(bbox)
        triangle_points =np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y+20],
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2 .FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["player"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                color=player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, bbox, color,track_id)  # Draw ellipse with red color 
                

            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255),track_id)
                # Draw ellipse with yellow color  

            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0),track_id)
                # Draw triangle with green color  

            output_video_frames.append(frame)

        return output_video_frames

