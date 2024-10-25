from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import os
import sys 
import cv2
import pandas as pd
from sklearn.cluster import KMeans
sys.path.append('.../')
from utils import get_center_of_the_box, get_width_of_the_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.team_colors = {}
        self.player_team_dict = {}  # player_id:team

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch
        return detections

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
        df_ball_positions=df_ball_positions.ffill().bfill()
        ball_positions=[{1:{"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 if non_player_cluster == 0 else 0
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player in player_detections.items():  # this have track_id:bbox in player_detection.items()
            bbox = player['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id
        return team_id

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
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

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

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

        # After processing all frames, assign team colors
        self.assign_team_color(frames[0], tracks["player"][0])  # Use the first frame to assign team colors
        
        # Assign team colors to all players in all frames
        for frame_num, frame in enumerate(frames):
            for track_id, player in tracks["player"][frame_num].items():
                team_id = self.get_player_team(frame, player['bbox'], track_id)
                tracks["player"][frame_num][track_id]['team_id'] = team_id
                tracks["player"][frame_num][track_id]['team_color'] = self.team_colors[team_id]

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
        y = int(bbox[1])
        x, _ = get_center_of_the_box(bbox)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks, team_possession):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["player"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                color = player.get("team_color", (0,0,255))  # Use assigned team color
                frame = self.draw_ellipse(frame, bbox, color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255), track_id)

            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255), track_id)

            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0), track_id)

            frame = self.draw_team_ball_control(frame, frame_num, team_possession)

            output_video_frames.append(frame)

        return output_video_frames
