from copy import deepcopy

import cv2
import numpy as np


def non_max_suppression_onnx(
    preds: np.ndarray,
    conf_thres: float = 0.25,
    iou_thresh: float = 0.45,
) -> np.ndarray:
    """Non-Maximum Suppression for ONNX predictions.

    Args:
        preds: Predictions [16, n_predictions] containing:
               [0:4]   - bbox (x_center, y_center, width, height)
               [4]     - objectness confidence
               [5:15]  - landmarks (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
               [15]    - class confidence
        conf_thres: Confidence threshold
        iou_thresh: IoU threshold for NMS

    Returns:
        Detections [n_det, 16] containing:
        [0:4]   - bbox (x1, y1, x2, y2)
        [4]     - confidence (objectness * class)
        [5:15]  - landmarks (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
        [15]    - class index
    """
    # Filter by objectness confidence (preds[4])
    obj_conf_mask = preds[4] > conf_thres
    detections = preds[:, obj_conf_mask]

    # Multiply objectness confidence by class confidence (detections[15:])
    detections[15:] *= detections[4:5]

    # Get best class confidence and index
    class_conf = detections[15:].max(axis=0, keepdims=True)
    class_idx = detections[15:].argmax(axis=0, keepdims=True).astype(float)

    # Convert bbox from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    bbox_xywh = detections[:4].T
    center_xy, size_wh = bbox_xywh[:, :2], bbox_xywh[:, 2:4]
    bbox_xyxy = np.concatenate([center_xy - size_wh / 2, center_xy + size_wh / 2], axis=1)

    # Concatenate: [bbox_xyxy, confidence, landmarks, class_idx]
    landmarks = detections[5:15].T
    detections = np.concatenate([bbox_xyxy, class_conf.T, landmarks, class_idx.T], axis=1)

    # Filter by class confidence
    detections = detections[class_conf.flatten() > conf_thres]

    if len(detections) == 0:
        return np.zeros((0, 16))

    # NMS - keep boxes with IoU below threshold
    boxes = detections[:, :4]
    scores = detections[:, 4]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        idx = order[0]
        keep_indices.append(idx)

        # Compute IoU between current box and remaining boxes
        inter_w = np.maximum(0, np.minimum(x2[idx], x2[order[1:]]) - np.maximum(x1[idx], x1[order[1:]]))
        inter_h = np.maximum(0, np.minimum(y2[idx], y2[order[1:]]) - np.maximum(y1[idx], y1[order[1:]]))
        inter_area = inter_w * inter_h
        iou = inter_area / (areas[idx] + areas[order[1:]] - inter_area)

        # Keep boxes with IoU <= threshold
        order = order[np.concatenate([[0], np.where(iou <= iou_thresh)[0] + 1])][1:]

    return detections[np.array(keep_indices, dtype=int)]


def rescale_coordinates(
    preds: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    # Rescale coordinates to original image
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    # Scale bounding boxes (x1, y1, x2, y2)
    preds[:, 0] *= scale_x  # x1
    preds[:, 1] *= scale_y  # y1
    preds[:, 2] *= scale_x  # x2
    preds[:, 3] *= scale_y  # y2

    # Scale landmarks (5 landmarks with x, y coordinates starting at index 5)
    for i in range(5):
        preds[:, 5 + 2 * i] *= scale_x  # landmark x
        preds[:, 5 + 2 * i + 1] *= scale_y  # landmark y

    return preds


def draw_bboxes_and_keypoints(
    img: np.ndarray,
    bboxes: np.ndarray,
    keypoints_all: np.ndarray,
    point_size: int = 5,
) -> np.ndarray:
    img = deepcopy(img)
    for i in range(len(bboxes)):
        # bbox is in (x1, y1, x2, y2) format
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox
        # keypoints is a flat array of 10 values: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        keypoints = keypoints_all[i]

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5,
        )

        cv2.circle(img, (keypoints[0], keypoints[1]), 2, (0, 0, 255), point_size)
        cv2.circle(img, (keypoints[2], keypoints[3]), 2, (0, 0, 255), point_size)
        cv2.circle(img, (keypoints[4], keypoints[5]), 2, (0, 0, 255), point_size)
        cv2.circle(img, (keypoints[6], keypoints[7]), 2, (0, 0, 255), point_size)
        cv2.circle(img, (keypoints[8], keypoints[9]), 2, (0, 0, 255), point_size)
    return img
