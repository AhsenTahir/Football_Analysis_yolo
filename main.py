from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_assigner import PlayerAssigner
from camera_movement_estimation import CameraMovementEstimator
def main():
    #reading the video
    video_frames = read_video("videos/1.mp4")
    tracker=Tracker('models/best_github.pt')
    
    tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/tracks.pkl')
    
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])

    camera_movement=camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement.pkl')
    
    tracks['ball']=tracker.ball_interpolation(tracks['ball'])



    team_assigner=TeamAssigner()

    team_assigner.assign_team_color(video_frames[0],tracks["player"][0])

    for frame_num,player_track in enumerate(tracks["player"]):
        for player_id,track in player_track.items():
           team=team_assigner.get_player_team(video_frames[frame_num],track["bbox"],player_id)
           track['players'][frame_num][player_id]['team']=team
           track['players'][frame_num][player_id]['team_color']=team_assigner.team_colors[team]

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

    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_possesion)

    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame=camera_movement)

   #save the video
    save_video(video_frames, "output_videos/output_video3.avi")

if __name__ == "__main__":
    main()    