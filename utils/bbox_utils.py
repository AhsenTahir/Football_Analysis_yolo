import math

def get_center_of_the_box(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width_of_the_box(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1

def measure_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def getFootPosition(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)
