from video_event_callback import VideoEventCallback
import math
from shapesimilarity import shape_similarity
import cv2
import numpy as np

class SwiftActivityCallback(VideoEventCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_positions = dict()

    def trigger_value(self, mask, box, category, name, track_id, detections):
        triggered = False

        if track_id in self.previous_positions:
            #Calculate speed
            print(f"Calculating speed for object {track_id}, of type: {name}, {category}")
            previous_box, previous_mask = self.previous_positions[track_id]
            external_speed = self.external_speed(box, previous_box, category)
            internal_speed = self.internal_speed(mask, previous_mask)
            speed = external_speed + internal_speed
            print(f"Internal speed: {internal_speed:.6f}, external speed: {external_speed:.6f}, total: {speed:.6f}, bound: {self.trigger_bound:.6f}")
            if speed > self.trigger_bound:
                triggered = True
        self.previous_positions[track_id] = (box, mask)
        return triggered
    
    def internal_speed(self, mask, previous_mask):
        mask_uint8 = mask.astype(np.uint8)
        previous_mask_uint8 = previous_mask.astype(np.uint8)
        contours_current, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_prev, _ = cv2.findContours(previous_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        similarity = 1
        if contours_current and contours_prev:
            current_contour = max(contours_current, key=cv2.contourArea)
            prev_contour = max(contours_prev, key=cv2.contourArea)
            # Squeeze contours to get arrays of points (shape: (num_points, 2))
            curve = current_contour.squeeze()
            previous_curve = prev_contour.squeeze()
            # Ensure curve is a list of [x, y] points (if squeeze returns a 1D array, fix it)
            if curve.ndim == 1:
                curve = np.expand_dims(curve, axis=0)
            if previous_curve.ndim == 1:
                previous_curve = np.expand_dims(previous_curve, axis=0)
            if len(curve) > 1 and len(previous_curve) > 1:
                similarity = shape_similarity(curve.tolist(), previous_curve.tolist(), checkRotation=False)
            else:
                print(f"Warning!: Object with no contour! {contours_current}, {contours_prev}")
        else:
            print(f"Warning!: Object with no contour! {contours_current}, {contours_prev}")
        return (1 - similarity) * self.fps
    
    def external_speed(self, box, previous_box, category):
        x1, y1, x2, y2 = box
        center = ((x1+x2)/2, (y1+y2)/2)
        u, v = center

        prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
        prev_center = ((prev_x1+prev_x2)/2, (prev_y1+prev_y2)/2)
        prev_u, prev_v = prev_center

        return self.distance(u, v, prev_u, prev_v, category) * self.fps
