from video_event_callback import VideoEventCallback

import math

class SwiftClosenessCallback(VideoEventCallback):
    def __init__(self, running_window_update_weight = 0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = running_window_update_weight
        self.distances = dict()

    def trigger(self, mask, box, category, name, track_id, detections):
        triggered = False
        for detection in detections:
            if detection.track_id != track_id:
                #Calculate distance between elements
                print(f"Calculating distance between object {track_id}, of type: {name}, {category}, and object {detection.track_id}, of type: {detection.category.name}, {detection.category.id}")
                x1, y1, x2, y2 = box
                center = ((x1+x2)/2, (y1+y2)/2)
                u, v = center

                other_x1, other_y1, other_x2, other_y2 = detection.bbox.get_shifted_box()
                #print(f"Box 1: {x1}, {x2}, {y1}, {y2}")
                #print(f"Box 2: {other_x1}, {other_x2}, {other_y1}, {other_y2}")
                other_center = ((other_x1+other_x2)/2, (other_y1+other_y2)/2)
                other_u, other_v = other_center
                if ((track_id, detection.track_id) in self.distances):
                    running_window_distance = self.distances[(track_id, detection.track_id)]
                    current_distance = self.distance(u, v, other_u, other_v, category)
                    self.distances[(track_id, detection.track_id)] = self.alpha * current_distance + (1 - self.alpha) * running_window_distance

                    triggered = triggered or (self.distances[(track_id, detection.track_id)] < self.trigger_bound and self.trigger_bound < running_window_distance)
                    print(f"Distance: {self.distances[(track_id, detection.track_id)]:.6f}, rolling_avg: {running_window_distance}, bound: {self.trigger_bound:.6f}, triggered {triggered}")
                else:
                    # Initialize distance between two objects
                    self.distances[(track_id, detection.track_id)] = self.distance(u, v, other_u, other_v, category)
        return triggered