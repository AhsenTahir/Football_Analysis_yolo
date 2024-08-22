from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
def main():
    #reading the video
    video_frames = read_video("videos/1.mp4")
    tracker=Tracker('models/best_github.pt')
    
    tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/tracks.pkl')
    tracks['ball']=tracker.ball_interpolation(tracks['ball'])



    team_assigner=TeamAssigner()

    team_assigner.assign_team_color(video_frames[0],tracks["player"][0])

    for frame_num,player_track in enumerate(tracks["player"]):
        for player_id,track in player_track.items():
           team=team_assigner.get_player_team(video_frames[frame_num],track["bbox"],player_id)
           track['players'][frame_num][player_id]['team']=team
           track['players'][frame_num][player_id]['team_color']=team_assigner.team_colors[team]
   #save the video
    save_video(video_frames, "output_videos/output_video3.avi")

if __name__ == "__main__":
    main()    