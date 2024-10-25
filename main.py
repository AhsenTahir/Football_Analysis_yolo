from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import os
import cv2
import sys
from sklearn.cluster import KMeans
import pandas as pd
import gc
from utils.video_utils import read_video, save_video
from utils.bbox_utils import measure_distance, getFootPosition, get_center_of_the_box, get_width_of_the_box
from tracker import Tracker
from team_assigner import TeamAssigner
from player_assigner.player_assigner import PlayerAssigner
from camera_movement_estimation import CameraMovementEstimator
import sys
sys.path.append('..')
from speed_and_distance_estimator import SpeedAnddistanceEstimator

def print_dict_structure(d, level=0):
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * level + str(key))
            print_dict_structure(value, level + 1)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            print('  ' * level + f'Index {i}')
            print_dict_structure(item, level + 1)
    else:
        print('  ' * level + str(d))

def main():
    tracks = None  # Initialize tracks to None
    try:
        print("Starting video processing...")
        #reading the video
        video_frames = read_video("videos/1.mp4")
        print(f"Number of input frames: {len(video_frames)}")
        tracker=Tracker('models/best_github.pt')
        
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='stubs/tracks.pkl')
        print(f"Number of frames in tracks: {len(tracks['player'])}")

        tracks = tracker.AddPositionsToTrack(tracks)
        camera_movement_estimator=CameraMovementEstimator(video_frames[0])

        camera_movement=camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=False,stub_path='stubs/camera_movement.pkl')
        camera_movement_estimator.add_adjust_position_to_track(tracks, camera_movement)

        tracks['ball']=tracker.ball_interpolation(tracks['ball'])

        # Add speed and distance calculation here, after all position data has been added
        speed_and_distance_estimator = SpeedAnddistanceEstimator()
        tracks = speed_and_distance_estimator.addSpeedAndDistance(tracks)

        # Add this debug print
        print("Sample player track after speed calculation:")
        if tracks["player"] and tracks["player"][0]:
            sample_player = next(iter(tracks["player"][0].values()))
            print(sample_player)

        team_assigner = TeamAssigner()

        team_assigner.assign_team_color(video_frames[0], tracks["player"][0])
        print_dict_structure(tracks)

        for frame_num, player_track in enumerate(tracks["player"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                tracks["player"][frame_num][player_id]['team'] = team
                tracks["player"][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        player_assigner=PlayerAssigner()
        team_possesion=[]
        for frame_num,player_track in enumerate(tracks["player"]):
            ball_bbox=tracks["ball"][frame_num][1]["bbox"]
            assigned_player=player_assigner.assign_ball_to_player(player_track,ball_bbox)

            if assigned_player!=-1:
                tracks["player"][frame_num][assigned_player]["has_ball"]=True
                team_possesion.append(tracks["player"][frame_num][assigned_player]["team"])
            else:
                team_possesion.append(-1)

        print("Starting to draw annotations...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_possesion)
        print(f"Drew annotations for {len(output_video_frames)} frames")
        
        if len(output_video_frames) == 0:
            raise ValueError("draw_annotations returned no frames")
        
        print(f"Number of frames after draw_annotations: {len(output_video_frames)}")

        output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame=camera_movement)
        print(f"Number of frames after draw_camera_movement: {len(output_video_frames)}")

        output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        print(f"Number of frames after draw_speed_and_distance: {len(output_video_frames)}")

        print("this is me drawing speed and distance lets check if its work or not ")
        print(tracks["player"][0])
        print(f"Number of frames: {len(output_video_frames)}")

        print(f"Final number of frames: {len(output_video_frames)}")
        
        if len(output_video_frames) > 0:
            print("Saving video...")
            save_video(output_video_frames, "output/output1.mp4")
            print("Video saved successfully")
        else:
            print("Error: No frames to save!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
