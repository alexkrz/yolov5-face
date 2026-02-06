from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.yolo import Model
from utils.general import non_max_suppression_face
from utils.postprocess import draw_bboxes_and_keypoints, non_max_suppression_onnx, rescale_coordinates

script_dir = Path(__file__).parent


@dataclass
class Config:
    img_fp = str(script_dir / "data" / "images" / "zidane.jpg")
    torch_fp = str(script_dir / "weights" / "yolov5n-face.pt")
    onnx_fp = str(script_dir / "weights" / "yolov5n-face.onnx")
    iou_thres = 0.45
    conf_thres = 0.25


def load_torch(args: Config):
    model = Model(cfg="models/yolov5n.yaml")
    # state_dict = model.state_dict()
    # print(state_dict.keys())
    weights = torch.load(args.torch_fp)
    model.load_state_dict(weights["model"].state_dict())

    img = Image.open(args.img_fp)
    tfms = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    input: torch.Tensor = tfms(img)
    input = input.unsqueeze(0)  # Add batch dimension
    print(input.shape)

    model.eval()
    outputs = model(input)
    preds_raw = outputs[0]  # Concatenated tensor
    print(preds_raw.shape)

    # Apply NMS
    preds = non_max_suppression_face(preds_raw, args.conf_thres, args.iou_thres)
    print(preds[0].shape)

    # NOTE: It looks like Unity prefers to do NMS along the last dimension


def load_onnx(args: Config):
    img = cv2.imread(args.img_fp)
    print("Original image shape:", img.shape)

    model = cv2.dnn.readNetFromONNX(args.onnx_fp)

    # Preprocess image for ONNX model
    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255.0,
        (640, 640),
        swapRB=True,
        crop=False,
    )
    print("Blob shape:", blob.shape)

    # Set input and run inference
    model.setInput(blob)
    outputs = model.forward()
    print("ONNX output shape:", outputs.shape)
    preds_raw = outputs[0]

    # Apply NMS
    preds = non_max_suppression_onnx(preds_raw, args.conf_thres, args.iou_thres)
    print("Predictions after NMS:", preds.shape)

    # Rescale coordinates
    preds = rescale_coordinates(preds, img)
    bboxes = np.array(preds[:, :4], dtype=int)
    keypoints_all = np.array(preds[:, 5:15], dtype=int)
    print("Bbox coordinates:", bboxes[0])

    img = draw_bboxes_and_keypoints(img, bboxes, keypoints_all)
    cv2.imwrite("results/result.jpg", img)


if __name__ == "__main__":
    args = Config
    # load_torch(args)
    load_onnx(args)
