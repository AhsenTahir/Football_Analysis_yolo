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

