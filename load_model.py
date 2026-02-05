from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.yolo import Model


def main(img_fp: str, model_fp: str):
    model = Model(cfg="models/yolov5n.yaml")
    # state_dict = model.state_dict()
    # print(state_dict.keys())
    weights = torch.load(model_fp)
    model.load_state_dict(weights["model"].state_dict())

    img = Image.open(img_fp)
    tfms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
    )
    input: torch.Tensor = tfms(img)
    input = input.unsqueeze(0)  # Add batch dimension
    print(input.shape)

    preds = model(input)[0]
    print(preds.shape)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    img_fp = script_dir / "data" / "images" / "zidane.jpg"
    model_fp = script_dir / "weights" / "yolov5n-face.pt"
    main(str(img_fp), str(model_fp))
