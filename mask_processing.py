import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import math

import argparse
import configparser

from swift_activity_callback import SwiftActivityCallback
from swift_closeness_callback import SwiftClosenessCallback
import sys
import matplotlib.pyplot as plt

# Defining classes to be compàtible with previous work using SAHI and Ultralytic's YOLOv8 libraries.
class Detection:
    class Mask:
        def __init__(self, mask):
            self.mask = mask
        def get_shifted_mask(self):
            return self.mask
    class Box:
        def __init__(self, box):
            self.box = box
        def get_shifted_box(self):
            return self.box
    class Category:
        def __init__(self, id, name):
            self.id = id
            self.name = name
    
    def __init__(self, mask, bbox, category_id, category_name, track_id):
        self.mask = Detection.Mask(mask)
        self.bbox = Detection.Box(bbox)
        self.category = Detection.Category(category_id, category_name)
        self.track_id = track_id

def process_video(subdir, files, object_masks_folder, class_masks_folder, model_dir, labels, palimage, output_dir,
                  frame_step=1, fps=30,
                  activity_trigger_length_sec=3, activity_frame_tolerance_sec=1, activity_trigger_bound=0.7,
                  closeness_trigger_length_sec=1, closeness_frame_tolerance_sec=1, closeness_trigger_bound=0.3,
                  camera_height_cm=30, camera_angle=-math.pi/4, focal_length=0.03, box_margin=5,
                  running_window_update_weight=0.4):
    
    # Calculate trigger lengths and tolerances in frames
    activity_trigger_length_frames = fps * activity_trigger_length_sec / frame_step
    activity_frame_tolerance_frames = fps * activity_frame_tolerance_sec / frame_step
    closeness_trigger_length_frames = fps * closeness_trigger_length_sec / frame_step
    closeness_frame_tolerance_frames = fps * closeness_frame_tolerance_sec / frame_step
    callback_fps = fps / frame_step

    activity_callback = SwiftActivityCallback(trigger_name="activity_callback",
                                              trigger_length=activity_trigger_length_frames,
                                              frame_tolerance=activity_frame_tolerance_frames,
                                              trigger_bound=activity_trigger_bound,
                                              output_path=(Path(output_dir) / os.path.basename(subdir)/"ActivityCallbackOutput/").expanduser(),
                                              snapshot_path=(Path(output_dir) / os.path.basename(subdir)/"ActivityCallbackOutput/Snapshots/").expanduser(),
                                              classification_model=model_dir,
                                              fps=callback_fps,
                                              camera_height_cm=camera_height_cm,
                                              camera_angle=camera_angle,
                                              focal_length=focal_length,
                                              box_margin=box_margin)
    closeness_callback = SwiftClosenessCallback(trigger_name="closeness_callback",
                                                trigger_length=closeness_trigger_length_frames,
                                                frame_tolerance=closeness_frame_tolerance_frames,
                                                trigger_bound=closeness_trigger_bound,
                                                output_path=(Path(output_dir) / os.path.basename(subdir)/"ClosenessCallbackOutput/").expanduser(),
                                                snapshot_path=(Path(output_dir) / os.path.basename(subdir)/"ClosenessCallbackOutput/Snapshots/").expanduser(),
                                                classification_model=model_dir,
                                                fps=callback_fps,
                                                camera_height_cm=camera_height_cm,
                                                camera_angle=camera_angle,
                                                focal_length=focal_length,
                                                box_margin=box_margin,
                                                running_window_update_weight=running_window_update_weight)
    i=0
    for file in files:
        if i % frame_step != 0:
            i+=1
            continue
        i+=1
        img = Image.open(os.path.join(subdir, file))
        img_np = np.array(img.convert("RGB"), dtype=np.uint8)
        obj_mask = Image.open(Path(object_masks_folder).expanduser() / Path(os.path.basename(subdir)) / (Path(file).stem + '.png'))
        cls_mask = Image.open(Path(class_masks_folder).expanduser() / Path(os.path.basename(subdir)) / (Path(file).stem + '.png'))
        print(f"Reading {object_masks_folder / Path(os.path.basename(subdir)) / (Path(file).stem + '.png')}")
        print(f"Reading {class_masks_folder / Path(os.path.basename(subdir)) / (Path(file).stem + '.png')}")
        obj_mask = obj_mask.quantize(dither=Image.NONE)
        cls_mask = cls_mask.convert("RGB")
        cls_mask = cls_mask.quantize(palette=palimage, dither=Image.NONE)
        np.set_printoptions(threshold=sys.maxsize)
        cls_palette = cls_mask.getpalette(rawmode="RGB")
        cls_palette = np.reshape(cls_palette, (-1, 3))

        detections = []
        for obj_id in np.unique(np.array(obj_mask)):
            mask = np.where(np.array(obj_mask) == obj_id, 255, 0).astype(np.uint8)
            mask_indices = np.nonzero(mask)
            cls_values = np.array(cls_mask)[mask_indices]
            cls_count = np.bincount(cls_values)
            cls = np.argmax(cls_count)
            cls = cls_palette[cls,:]

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, x + w, y + h)
            
            cv2.drawContours(img_np, contours, -1, tuple(map(int, cls)), 3)
            split_rgb = [s.split(',') for s in labels["color_rgb"]]
            rgb_array = np.array(split_rgb, dtype=int)
            label_mask = np.all(rgb_array == cls, axis=1)
            first_matching_row_index = min(labels.loc[label_mask, :].index)
            label=labels.loc[first_matching_row_index,:]
            if label.name != 5:
                detections.append(Detection(mask, bbox, label.name, label["# label"], obj_id))
        activity_callback.callback(Path(file).stem, img_np, detections)
        closeness_callback.callback(Path(file).stem, img_np, detections)

def main():
    # --- Default Configuration ---
    config_defaults = {
        'Processing': {
            'fps': '30',
            'frame_step': '1',
        },
        'Camera': {
            'camera_height_cm': '30',
            'camera_angle': str(-math.pi/4),
            'focal_length': '0.03',
            'box_margin': '5',
        },
        'ActivityCallback': {
            'trigger_length_sec': '3',
            'frame_tolerance_sec': '1',
            'trigger_bound': '0.7',
        },
        'ClosenessCallback': {
            'trigger_length_sec': '1',
            'frame_tolerance_sec': '1',
            'trigger_bound': '0.3',
            'running_window_update_weight': '0.4',
        }
    }

    # --- Read Config File ---
    config = configparser.ConfigParser()
    # Set defaults first
    config.read_dict(config_defaults)
    
    # Create a preliminary parser just to get the config file path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to configuration file (.ini format)")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if pre_args.config and Path(pre_args.config).exists():
        print(f"Reading configuration from: {pre_args.config}")
        config.read(pre_args.config)
    elif pre_args.config:
        print(f"Warning: Config file specified but not found: {pre_args.config}. Using defaults.")

    # --- Setup Main Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Process video frames with segmentation masks to find events.",
        parents=[pre_parser], # Inherit --config argument
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    
    # --- Positional Arguments ---
    parser.add_argument("video_path", type=str, help="Path to the input video frames directory. Can contain subdirectories for several videos.")
    parser.add_argument("masks_path", type=str, help="Path to the input video masks directory. Must contain 'Objects' and 'Classes' subdirectories.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    
    # --- Optional Model Argument ---
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the event classification model (e.g., Hugging Face VideoMAE). If not provided, classification will be disabled.")

    # --- Optional Arguments (with defaults from config/hardcoded) ---
    # Processing
    parser.add_argument("--fps", type=int,
                        default=config.getint('Processing', 'fps'),
                        help="Frames per second of the input video.")
    parser.add_argument("--frame_step", type=int,
                        default=config.getint('Processing', 'frame_step'),
                        help="Process every Nth frame.")
    
    # Camera Parameters
    parser.add_argument("--camera_height_cm", type=float,
                        default=config.getfloat('Camera', 'camera_height_cm'),
                        help="Camera height in centimeters.")
    parser.add_argument("--camera_angle", type=float,
                        default=config.getfloat('Camera', 'camera_angle'),
                        help="Camera angle in radians (negative for downward tilt).")
    parser.add_argument("--focal_length", type=float,
                        default=config.getfloat('Camera', 'focal_length'),
                        help="Camera focal length in meters.")
    parser.add_argument("--box_margin", type=int,
                        default=config.getint('Camera', 'box_margin'),
                        help="Margin around bounding boxes in pixels.")
    
    # Activity Callback
    parser.add_argument("--activity_trigger_length_sec", type=float,
                        default=config.getfloat('ActivityCallback', 'trigger_length_sec'),
                        help="Activity trigger length in seconds.")
    parser.add_argument("--activity_frame_tolerance_sec", type=float,
                        default=config.getfloat('ActivityCallback', 'frame_tolerance_sec'),
                        help="Activity frame tolerance in seconds.")
    parser.add_argument("--activity_trigger_bound", type=float,
                        default=config.getfloat('ActivityCallback', 'trigger_bound'),
                        help="Activity trigger bound (threshold).")

    # Closeness Callback
    parser.add_argument("--closeness_trigger_length_sec", type=float,
                        default=config.getfloat('ClosenessCallback', 'trigger_length_sec'),
                        help="Closeness trigger length in seconds.")
    parser.add_argument("--closeness_frame_tolerance_sec", type=float,
                        default=config.getfloat('ClosenessCallback', 'frame_tolerance_sec'),
                        help="Closeness frame tolerance in seconds.")
    parser.add_argument("--closeness_trigger_bound", type=float,
                        default=config.getfloat('ClosenessCallback', 'trigger_bound'),
                        help="Closeness trigger bound (threshold).")
    parser.add_argument("--running_window_update_weight", type=float,
                        default=config.getfloat('ClosenessCallback', 'running_window_update_weight'),
                        help="Weight for updating the running window in closeness calculations.")

    # --- Parse Arguments ---
    # Use remaining_argv so arguments are not parsed twice (by pre_parser and parser)
    args = parser.parse_args(remaining_argv)

    # --- Assign Variables ---
    video_folder = Path(args.video_path)
    object_masks_folder = Path(args.masks_path) / "Objects/"
    class_masks_folder = Path(args.masks_path) / "Classes/"
    model_folder = Path(args.model_path) if args.model_path else None
    output_folder = Path(args.output_path)
    
    # Get parameters, CLI args override config/defaults
    fps = args.fps
    frame_step = args.frame_step
    
    # Camera parameters
    camera_height_cm = args.camera_height_cm
    camera_angle = args.camera_angle
    focal_length = args.focal_length
    box_margin = args.box_margin
    
    # Activity callback parameters
    activity_trigger_length_sec = args.activity_trigger_length_sec
    activity_frame_tolerance_sec = args.activity_frame_tolerance_sec
    activity_trigger_bound = args.activity_trigger_bound
    
    # Closeness callback parameters
    closeness_trigger_length_sec = args.closeness_trigger_length_sec
    closeness_frame_tolerance_sec = args.closeness_frame_tolerance_sec
    closeness_trigger_bound = args.closeness_trigger_bound
    running_window_update_weight = args.running_window_update_weight

    # --- Load Labels ---
    # Assuming labelmap.txt is one level up from the video_folder
    label_map_path = video_folder.parent / "labelmap.txt"
    if not label_map_path.exists():
        print(f"Error: labelmap.txt not found at expected location: {label_map_path}")
        sys.exit(1) # Exit if labelmap is crucial and not found
    labels = pd.read_csv(label_map_path, sep=':')
    split_rgb = [s.split(',') for s in labels["color_rgb"]]
    rgb_array = np.array(split_rgb, dtype=int).flatten()
    
    palettedata = list(rgb_array)
    # Fill the entire palette so that no entries in Pillow's
    # default palette for P images can interfere with conversion
    NUM_ENTRIES_IN_PILLOW_PALETTE = 256
    num_bands = len("RGB")
    num_entries_in_palettedata = len(palettedata) // num_bands
    palettedata.extend([0, 0, 0] * (NUM_ENTRIES_IN_PILLOW_PALETTE - num_entries_in_palettedata))
    
    # Create a palette image whose size does not matter
    arbitrary_size = 16, 16
    palimage = Image.new('P', arbitrary_size)
    palimage.putpalette(palettedata)

    try:
        for subdir, dirs, files in os.walk(os.path.expanduser(video_folder), followlinks=True):
           process_video(subdir, files, object_masks_folder, class_masks_folder, model_folder, labels, palimage, output_folder,
                         frame_step=frame_step, fps=fps,
                         activity_trigger_length_sec=activity_trigger_length_sec,
                         activity_frame_tolerance_sec=activity_frame_tolerance_sec,
                         activity_trigger_bound=activity_trigger_bound,
                         closeness_trigger_length_sec=closeness_trigger_length_sec,
                         closeness_frame_tolerance_sec=closeness_frame_tolerance_sec,
                         closeness_trigger_bound=closeness_trigger_bound,
                         camera_height_cm=camera_height_cm,
                         camera_angle=camera_angle,
                         focal_length=focal_length,
                         box_margin=box_margin,
                         running_window_update_weight=running_window_update_weight)
    except Exception as e:
        print(f"Error processing directory {subdir}: {e}")
        # Optionally re-raise or handle more gracefully
        # raise e

if __name__ == "__main__":
    main()
