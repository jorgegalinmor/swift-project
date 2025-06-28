
import cv2
import torch
from pathlib import Path
from typing import List, Optional

import copy

import logging
logger = logging.getLogger(__name__)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

from ultralytics.utils.files import increment_path

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from matplotlib import pyplot as plt

import numpy as np

def perform_inference_tracking(self, image: np.ndarray):
    """
    Prediction is performed using self.model and the prediction result is set to self._original_predictions.
    If predictions have masks, each prediction is a tuple like (boxes, masks).
    Args:
        image: np.ndarray
            A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.

    """

    from ultralytics.engine.results import Masks

    # Confirm model is loaded
    if self.model is None:
        raise ValueError("Model is not loaded, load it by calling .load_model()")
    if not hasattr(self, 'track_model'):
        self.track_model = copy.deepcopy(self.model)

    kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}

    if self.image_size is not None:
        kwargs = {"imgsz": self.image_size, **kwargs}

    if image.shape[0] == self.track_frame_height and image.shape[1] == self.track_frame_width: # Do tracking on the full image
        prediction_result = self.track_model.track(image[:, :, ::-1], **kwargs, persist=True)  # YOLOv8 expects numpy arrays to have BGR
    else:
        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLOv8 expects numpy arrays to have BGR

    self._results = prediction_result
    self._original_shape = image.shape
    if self.has_mask:
        if not prediction_result[0].masks:
            prediction_result[0].masks = Masks(
                torch.tensor([], device=self.model.device), prediction_result[0].boxes.orig_shape
            )

        # We do not filter results again as confidence threshold is already applied above
        self._original_predictions  = [
            (
                result.boxes.data,
                result.masks.data,
                result.boxes.id.int().cpu().tolist() if result.boxes.id != None else None
            )
            for result in prediction_result
        ]

    else:  # If model doesn't do segmentation then no need to check masks
        # We do not filter results again as confidence threshold is already applied above
        self._original_predictions  = [result.boxes.data for result in prediction_result]

def create_object_prediction_list_from_original_predictions_tracking(
    self,
    shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
    full_shape_list: Optional[List[List[int]]] = None,
):
    """
    self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
    self._object_prediction_list_per_image.
    Args:
        shift_amount_list: list of list
            To shift the box and mask predictions from sliced image to full sized image, should
            be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
        full_shape_list: list of list
            Size of the full image after shifting, should be in the form of
            List[[height, width],[height, width],...]
    """
    original_predictions = self._original_predictions

    # compatilibty for sahi v0.8.15
    shift_amount_list = fix_shift_amount_list(shift_amount_list)
    full_shape_list = fix_full_shape_list(full_shape_list)
    # handle all predictions
    object_prediction_list_per_image = []
    for image_ind, image_predictions in enumerate(original_predictions):
        shift_amount = shift_amount_list[image_ind]
        full_shape = None if full_shape_list is None else full_shape_list[image_ind]
        object_prediction_list = []
        if self.has_mask:
            image_predictions_in_xyxy_format = image_predictions[0]
            image_predictions_masks = image_predictions[1]
            image_predictions_track_ids = image_predictions[2]
            for prediction, bool_mask, track_id in zip(
                image_predictions_in_xyxy_format.cpu().detach().numpy(),
                image_predictions_masks.cpu().detach().numpy(),
                image_predictions_track_ids if image_predictions_track_ids else [None] * len(image_predictions_in_xyxy_format),
            ):
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]
                if track_id != None:
                    category_name = f"{category_name}_{track_id}"

                orig_width = self._original_shape[1]
                orig_height = self._original_shape[0]
                bool_mask = cv2.resize(bool_mask.astype(np.uint8), (orig_width, orig_height))
                segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                if len(segmentation) == 0:
                    continue
                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction.track_id = track_id
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        else:  # Only bounding boxes
            # process predictions
            for prediction in image_predictions.data.cpu().detach().numpy():
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

    self._object_prediction_list_per_image = object_prediction_list_per_image

def sahi_run(weights="yolov8n.pt", source="test.mp4", view_img=False, save_img=False, exist_ok=False, out_folder="ultralytics_results_with_sahi_segment", track = False, callbacks = []):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = weights
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
    )
    if track:
        detection_model.track_frame_width = frame_width
        detection_model.track_frame_height = frame_height
        detection_model.perform_inference = perform_inference_tracking.__get__(detection_model, AutoDetectionModel)
        detection_model._create_object_prediction_list_from_original_predictions = create_object_prediction_list_from_original_predictions_tracking.__get__(detection_model, AutoDetectionModel)

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path(out_folder) / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = (
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
            )
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(
                frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
            )
            cv2.putText(
                frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            )

        if view_img:
            #cv2.imshow(Path(source).stem, frame)
            plt.imshow(frame)
            plt.show()
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()



def sahi_run_segment(weights="yolov8n.pt", source="test.mp4", view_img=False, save_img=False, exist_ok=False, out_folder="ultralytics_results_with_sahi_segment", track = False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    yolov8_model_path = weights
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
    )
    if track:
        detection_model.track_frame_width = frame_width
        detection_model.track_frame_height = frame_height
        detection_model.perform_inference = perform_inference_tracking.__get__(detection_model, AutoDetectionModel)
        detection_model._create_object_prediction_list_from_original_predictions = create_object_prediction_list_from_original_predictions_tracking.__get__(detection_model, AutoDetectionModel)


    # Output setup
    save_dir = increment_path(Path(out_folder) / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
    #out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        time_ms = videocapture.get(cv2.CAP_PROP_POS_MSEC)
        if not success:
            break

        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []
        # for ind, _ in enumerate(object_prediction_list):
        #     boxes = (
        #         object_prediction_list[ind].bbox.minx,
        #         object_prediction_list[ind].bbox.miny,
        #         object_prediction_list[ind].bbox.maxx,
        #         object_prediction_list[ind].bbox.maxy,
        #     )
        #     clss = object_prediction_list[ind].category.name
        #     boxes_list.append(boxes)
        #     clss_list.append(clss)

        # for box, cls in zip(boxes_list, clss_list):
            # x1, y1, x2, y2 = box
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            # label = str(cls)
            # t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            # cv2.rectangle(
            #     frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
            # )
            # cv2.putText(
            #     frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            # )

        annotator = Annotator(frame, line_width=2)

        for ind, _ in enumerate(object_prediction_list):
            mask = object_prediction_list[ind].mask
            category = object_prediction_list[ind].category.id
            name = object_prediction_list[ind].category.name
            if mask is not None:
                for segment in mask.segmentation:
                    annotator.seg_bbox(mask=np.array(segment, dtype=np.int32).reshape(-1,2), mask_color=colors(category, True), label=str(name))

        video_writer.write(frame)

        if view_img:
            #cv2.imshow(Path(source).stem, frame)
            plt.imshow(frame)
            plt.show()
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()