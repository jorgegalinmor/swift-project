from ultralytics.utils.files import increment_path
from pathlib import Path
import math
from swift_metrics import adult_swift_height, chick_swift_height, fledgeling_swift_height
import shutil
import os
from transformers import pipeline
import json

import cv2

class VideoEventCallback:
    def __init__(self, trigger_name, trigger_length=60, trigger_bound=0.5, frame_tolerance=5, 
                 output_path=None, snapshot_path = None, fps=30,
                 classification_model = None, classification_model_type = "hugging-face-video",
                 camera_height_cm = 30, camera_angle = -math.pi/4, focal_length = 0.03, box_margin = 5):
        self.trigger_name = trigger_name
        self.trigger_length = trigger_length
        self.trigger_bound = trigger_bound
        self.frame_tolerance = frame_tolerance
        self.current_detections = dict()
        self.output_path = increment_path(Path(output_path), True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = snapshot_path
        self.classification_model_type = classification_model_type
        self.classifier = None
        self.camera_height_cm = camera_height_cm
        self.camera_angle = camera_angle
        self.focal_length = focal_length
        self.box_margin = box_margin
        self.fps = fps

        if self.output_path != None:
            #Save detection
            save_file = increment_path(Path(self.output_path) / (self.trigger_name + "_detection_list.txt"), True)
            save_file.write_text(f"start_frame,end_frame,names,ids,classification\n")
        if self.snapshot_path != None:
            os.makedirs(self.snapshot_path, exist_ok=True)
            shutil.rmtree(self.snapshot_path)
            os.makedirs(self.snapshot_path, exist_ok=True)

        if (classification_model != None) and (classification_model_type == "hugging-face-video"):
            self.classifier = pipeline(task="video-classification", model=classification_model)
            self.classifier_type = "video"
        

    def __del__(self):
        #Save current detections
        for track_id in self.current_detections:
            print(f"{track_id} has {self.current_detections[track_id]['frame_count']} detections, it needs at least {self.trigger_length}-{self.frame_tolerance}")
            if self.current_detections[track_id]["frame_count"] > self.trigger_length-self.frame_tolerance:
                self.save_detection(self.current_detections[track_id])

    def callback(self, frame_fn, frame, detections):
        self.height = len(frame)
        self.width = len(frame[0])
        for track_id in self.current_detections:
            self.current_detections[track_id]["detections"].append((frame_fn, frame))
        for ind, _ in enumerate(detections):
            mask = detections[ind].mask.get_shifted_mask()
            category = detections[ind].category.id
            name = detections[ind].category.name
            track_id = detections[ind].track_id

            box = detections[ind].bbox.get_shifted_box()

            if track_id != None:
                if self.trigger(mask, box, category, name, track_id, detections):
                    box, names, track_ids = self.include_adjacent(box, detections)
                    names.add(name)
                    track_ids.add(track_id)
                    #New detection
                    if track_id not in self.current_detections:
                        print(f"New detection for id {track_id}")
                        print(box)
                        self.current_detections[track_id] = {
                            "detections": [(frame_fn, frame)],
                            "box": box,
                            "start_frame": frame_fn,
                            "end_frame": -1,
                            "category": category,
                            "names": names,
                            "track_ids": track_ids,
                            "frame_count": 1,
                            "missed_frames": 0
                        }
                    else:
                        #Detection tracking
                        print(f"Id {track_id} has {self.current_detections[track_id]['frame_count']+1} detections, it needs at least {self.trigger_length}")
                        prev_box = self.current_detections[track_id]["box"]
                        print(prev_box)
                        self.current_detections[track_id]["box"] = (min(prev_box[0], box[0]),
                                                                    min(prev_box[1], box[1]),
                                                                    max(prev_box[2], box[2]),
                                                                    max(prev_box[3], box[3]))
                        print(self.current_detections[track_id]["box"])
                        self.current_detections[track_id]["frame_count"] += 1
                        self.current_detections[track_id]["missed_frames"] = 0
                elif track_id in self.current_detections:
                    print(f"Id {track_id} has {self.current_detections[track_id]['frame_count']+1} detections, it needs at least {self.trigger_length}, it has missed {self.current_detections[track_id]['missed_frames']+1} detections, with a tolerance of  {self.frame_tolerance}.")
                    #Missed detection
                    prev_box = self.current_detections[track_id]["box"]
                    print(prev_box)
                    self.current_detections[track_id]["box"] = (min(prev_box[0], box[0]),
                                                                min(prev_box[1], box[1]),
                                                                max(prev_box[2], box[2]),
                                                                max(prev_box[3], box[3]))
                    print(self.current_detections[track_id]["box"])
                    self.current_detections[track_id]["frame_count"] += 1
                    self.current_detections[track_id]["missed_frames"] += 1
                    if self.current_detections[track_id]["missed_frames"] > self.frame_tolerance:
                        print(f"Id {track_id} has lost the detection.")
                        #Lost detection
                        self.current_detections[track_id]["end_frame"] = frame_fn
                        if self.current_detections[track_id]["frame_count"] > self.trigger_length:
                            self.save_detection(self.current_detections[track_id])
                        del self.current_detections[track_id]
    
    def include_adjacent(self, box, detections):
        min_u1, min_v1, max_u1, max_v1 = box
        min_u1, min_v1, max_u1, max_v1 = min_u1 - self.box_margin, min_v1 - self.box_margin, max_u1 + self.box_margin, max_v1 + self.box_margin
        to_add = [box]
        names = set()
        track_ids = set()
        for ind, _ in enumerate(detections):
            box2 = detections[ind].bbox.get_shifted_box()
            min_u2, min_v2, max_u2, max_v2 = box2
            min_u2, min_v2, max_u2, max_v2 = min_u2 - self.box_margin, min_v2 - self.box_margin, max_u2 + self.box_margin, max_v2 + self.box_margin

            if not (max_u1 < min_u2 or max_u2 < min_u1 or max_v1 < min_v2 or max_v2 < min_v1): # if intersects:
                to_add.append(box2)
                names.add(detections[ind].category.name)
                track_ids.add(detections[ind].track_id)
        union_box = (
            min(min_u for min_u,_,_,_ in to_add) - self.box_margin,
            min(min_v for _,min_v,_,_ in to_add) - self.box_margin,
            max(max_u for _,_,max_u,_ in to_add) + self.box_margin,
            max(max_v for _,_,_,max_v in to_add) + self.box_margin,
        )

        return union_box, names, track_ids

    def crop_frame(self, frame, box):
        min_u, min_v, max_u, max_v = box
        print(f"Cropping frame between ({min_v}, {min_u}) and ({max_v}, {max_u}), frame dim: ({frame.shape})")
        result = frame[max(0,min_v):min(frame.shape[0], max_v), max(0, min_u):min(frame.shape[1], max_u), :]
        print(f"Result shape: {result.shape}")
        return result

    def save_detection(self, detection):
        classification = None
        if self.classifier != None and self.classifier_type == "image":
            classification = self.classify_frames(detection["detections"])
        if self.snapshot_path != None:
            box = detection["box"]
            box = (max(0,box[0]), max(0,box[1]), min(self.width,box[2]), min(self.height,box[3]))
            if (box[3]-box[1]) % 2 == 1:
                box = list(box)
                box[3] -= 1 
                box = tuple(box)
            if (box[2]-box[0]) % 2 == 1:
                box = list(box)
                box[2] -= 1 
                box = tuple(box)
            print(box)
            if (box[0] < box[2]) and (box[1] < box[3]):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                #Save snapshot
                output_fn = increment_path(Path(self.snapshot_path) / f"{self.trigger_name}_{detection['start_frame']}_{detection['end_frame']}_{classification}_snapshots" / f"video.avi", True)
                output_fn.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(str(output_fn), fourcc, fps=self.fps, frameSize=(box[2]-box[0], box[3]-box[1]))
                if (self.classifier != None) and (self.classifier_type == "video") and (self.classification_model_type == "hugging-face-video"):
                    cls_output_fn = increment_path(Path(self.snapshot_path) / f"{self.trigger_name}_{detection['start_frame']}_{detection['end_frame']}_{classification}_snapshots" / f"cls_video.avi", True)
                    cls_output_fn.parent.mkdir(parents=True, exist_ok=True)
                    cls_img_size = self.classifier.model.config.image_size
                    cls_video_writer = cv2.VideoWriter(str(cls_output_fn), fourcc, fps=self.fps, frameSize=(cls_img_size, cls_img_size))
                for fn, detect in detection["detections"]:
                    save_file = increment_path(Path(self.snapshot_path) / f"{self.trigger_name}_{detection['start_frame']}_{detection['end_frame']}_{classification}_snapshots" / f"{fn}.jpg", True)
                    print(f"Saving image with shape {detect.shape} at {save_file}.")
                    crp_img = self.crop_frame(detect, box)
                    cv2.imwrite(str(save_file), crp_img)
                    video_writer.write(crp_img)
                    cls_video_writer.write(cv2.resize(crp_img, (cls_img_size, cls_img_size)))
                video_writer.release()
                required_frames = self.classifier.model.config.num_frames
                num_frames = len(detection["detections"])
                if num_frames < required_frames:
                    print(f"Padding video with {required_frames - num_frames} duplicate frames.")
                    for _ in range(required_frames - num_frames):
                        cls_video_writer.write(cv2.resize(crp_img, (cls_img_size, cls_img_size))) # Append copies of the last frame
                cls_video_writer.release()
                if (self.classifier != None) and (self.classifier_type == "video") and (self.classification_model_type == "hugging-face-video"):
                    classification = self.classifier(str(cls_output_fn))

                    # Find a unique name
                    counter = 1
                    snapshot_dir = Path(self.snapshot_path) / f"{self.trigger_name}_{detection['start_frame']}_{detection['end_frame']}_{classification[0]['label']}_snapshots"
                    new_snapshot_dir = snapshot_dir
                    while new_snapshot_dir.exists():
                        new_snapshot_dir = snapshot_dir.parent / f"{snapshot_dir.name}_{counter}"
                        counter += 1
                    output_fn.parent.rename(new_snapshot_dir)
        
        if self.output_path is not None:
            #Save detection
            save_file = increment_path(Path(self.output_path) / (self.trigger_name + "_detection_list.txt"), True)
            with open(save_file, "a", encoding="utf-8") as f:
                f.write(f"{detection['start_frame']}, {detection['end_frame']}, {json.dumps(list(detection['names']))}, {json.dumps(list(detection['track_ids']), default=lambda x: int(x))}, {json.dumps(classification)})\n")

    def classify_frames(self, detections):
        frame_classifications = [self.classifier(frame) for _, frame in detections]
        n_detected_frames = len(frame_classifications)
        counts = dict()
        for clas in frame_classifications:
            counts[clas] = counts.get(clas, 0) + 1
        classification = ""
        for clas in counts:
            classification += f"{counts[clas] / n_detected_frames} {clas}"
        return classification

    def trigger_value(self, mask, box, category, name, track_id, detections):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def trigger(self, mask, box, category, name, track_id, detections):
        return self.trigger_value(mask, box, category, name, track_id, detections) > self.trigger_bound
    

    def worldPosition(self, u, v, x = None, y = None, z = None, distance = None):
        theta = self.camera_angle
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        h=self.height #Image height
        w=self.width # Image width
        f=self.focal_length
        ty= self.camera_height_cm / 100 #cm to m
        # Aspect ratio is 16:9, hence sensor might be 1/2.3'' or 1/3'':
        sensor_width = 0.006 # rough avg between 6.17 and 4.8 mm
        sensor_height = 0.004 # rough avg between 4.55 and 3.6 mm
        u_centered = (u - w / 2) * (sensor_width / w) # Horizontal shift
        v_centered = (v - h / 2) * (sensor_height / h)  # Vertical shift

        ray_x = u_centered
        ray_y = v_centered * cos_th + f * sin_th
        ray_z = f * cos_th - v_centered * sin_th

        if y is not None:
            lambda_ = (y - ty) / ray_y
            x = lambda_ * ray_x
            z = lambda_ * ray_z
        elif x is not None:
            lambda_ = x / ray_x
            y = ty + lambda_ * ray_y
            z = lambda_ * ray_z
        elif z is not None:
            lambda_ = z / ray_z
            x = lambda_ * ray_x
            y = ty + lambda_ * ray_y
        elif distance is not None:
            norm = math.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
            lambda_ = distance / norm
            x = lambda_ * ray_x
            y = ty + lambda_ * ray_y
            z = lambda_ * ray_z
        return (x,y,z)

    def distance(self, u1, v1, u2, v2, category):
        #print(f"Calculating distance between ({u1}, {v1}) and ({u2}, {v2}).")
        y = adult_swift_height
        prev_y = adult_swift_height
        if category == "chick_swift":
            y = chick_swift_height/2
            prev_y = chick_swift_height/2
        if category == "fledgeling_swift_chick":
            y = fledgeling_swift_height
            prev_y = fledgeling_swift_height

        X1, Y1, Z1 = self.worldPosition(u1, v1, y=y)
        X2, Y2, Z2 = self.worldPosition(u2, v2, y=prev_y)

        distance = math.sqrt((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2)
        #print(f"World coordinates: ({X1}, {Y1}, {Z1}) and ({X2}, {Y2}, {Z2}), distance {distance:.6f}.")
        return distance