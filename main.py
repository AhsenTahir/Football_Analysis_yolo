from utils import read_video, save_video


def main():
    #reading the video
    video_frames = read_video("videos/1.mp4")
    
    
   #save the video
    save_video(video_frames, "output_videos/output_video3.avi")

if __name__ == "__main__":
    main()    