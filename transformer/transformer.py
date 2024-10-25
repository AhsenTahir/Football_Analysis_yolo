import cv2
import numpy as np

class ViewTransformer():
    def __init__(self):
        court_width=68
        court_length=23.32

        self.pixel_Verticies=np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
            ])
        self.target_Verticies=np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]
            ])
        self.pixel_Verticies=self.pixel_Verticies.astype(np.float32)
        self.target_Verticies=self.target_Verticies.astype(np.float32)

        self.perspective_transformer=cv2.getPerspectiveTransform(self.pixel_Verticies,self.target_Verticies)
    def transform_point(self,point):
        p=(int(point[0]),int(point[1]))
        isInside=cv2.pointPolygonTest(self.pixel_Verticies,p,False)
        if isInside<=0:
            return None
        reshaped_point=point.reshape(-1,1,2).astype(np.float32)
        transformed_point=cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return transformed_point

    def add_transformed_position_to_tracks(self,tracks):
        for object,objects_track in tracks.items():
            for frame_num,frame in enumerate(objects_track):
                for track_id,track in frame.items():
                    position=track["position_adjusted"]
                    transformed_position=self.transform_point(position)
                    track[object][frame_num][track_id]["transformed_position"]=transformed_position
                    if position_transformed is not None:
                       position_transformed=transformed_position.squeeze().tolist()
                       tracks[object][frame_num][track_id]["transformed_position"]=position_transformed
        return tracks