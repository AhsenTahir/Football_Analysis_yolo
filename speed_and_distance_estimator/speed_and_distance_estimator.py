import sys
sys.path.append('..')
from utils import measure_distance,getFootPosition
import cv2
class SpeedAnddistanceEstimator():
    def __init__(self):
        self.frameWindow=5
        self.frameRate=24

    def addSpeedAndDistance(self,tracks):
            total_distance=[]
            # Add speed and distance to the frame
            for object,object_tracks in tracks.items():
                if object=='ball' or object=='referees':#only be checking the speed of players
                    continue
                numberOfFrames=len(object_tracks)
                for frame_num in range(0,numberOfFrames,self.frameWindow):
                    last_frame=min(frame_num+self.frameWindow,numberOfFrames-1)
                    for track_id,_ in object_tracks[frame_num].items():
                        if track_id not in object_tracks[last_frame]:
                            continue
                        start_position=object_tracks[frame_num][track_id]['position_transformed']
                        end_position=object_tracks[last_frame][track_id]['position_transformed']

                        if start_position is None or end_position is None:
                            continue
                        distance_convered=measure_distance(start_position,end_position)
                        time_elapsed=(last_frame-frame_num)/self.frameRate
                        speed=distance_convered/time_elapsed
                        speed_in_kmph=speed*3.6
                        # object_tracks[last_frame][track_id]['speed']=speed_in_kmph

                        if object not in total_distance:
                            total_distance[object]={}

                        if track_id not in total_distance[object]:
                            total_distance[object][track_id]=0

                        total_distance[object][track_id]+=distance_convered

                        for frame_num_batch in range(frame_num,last_frame):
                            if track_id not in object_tracks[frame_num_batch]:
                                continue
                            object_tracks[frame_num_batch][track_id]['speed']=speed_in_kmph
                            object_tracks[frame_num_batch][track_id]['distance']=total_distance[object][track_id]

            
                    


    def draw_speed_and_distance(self,frames,tracks):
            output_frames=[]
            for frame_num,frame in enumerate(frames):
                for object ,object_tracks in tracks.item():
                    if object=="ball" or object=="referees":
                        continue
                    for track_id,track in object_tracks[frame_num].items():
                        if "speed" in track:
                            speed=track.get('speed',None)
                            distance=track.get('distance',None)
                            if speed is None or distance is None :
                                continue
                            bbox=track['bbox']
                            position=getFootPosition(bbox)
                            position=list(position)
                            position[1]+=40

                            position=tuple(map(int,position))
                            cv2.putText(frame,f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                            cv2.putText(frame,f"{speed:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

                            output_frames.append(frame)

            return output_frames
