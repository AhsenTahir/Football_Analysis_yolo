import sys
sys.path.append('.../')
from utils import get_center_of_the_box, measure_distance

class PlayerAssigner():
    def __init__(self):
        self.maximum_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_the_box(ball_bbox)
        minimum_distance = 99999999
        assigned_player = -1
        for player_id, player in players.items():
            player_position = get_center_of_the_box(player['bbox'])
            distance_of_left_foot = measure_distance((player['bbox'][0], player['bbox'][-1]), ball_position)
            distance_of_right_foot = measure_distance((player['bbox'][2], player['bbox'][-1]), ball_position)
            distance = min(distance_of_left_foot, distance_of_right_foot)
            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
