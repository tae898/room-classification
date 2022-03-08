import io
import logging

import albumentations as A
import jsonpickle
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from flask import Flask, request
from PIL import Image
from torch import nn

from train import RoomEfficientNet
from utils import crop_center_square

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask(__name__)

image_size = 300
batch_size = 64
num_classes = 7
efficientnet = "efficientnet-b3"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

net = RoomEfficientNet.load_from_checkpoint(
    checkpoint_path="model.ckpt",
    num_classes=num_classes,
    efficientnet=efficientnet,
    weights_path="efficientnet-b3-5fb5a3c3.pth",
)
net.to(device)
net.eval()
net.freeze()
app.logger.debug("model loaded!")

transform = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ]
)

label2idx = {
    "interior": 0,
    "bathroom": 1,
    "bedroom": 2,
    "exterior": 3,
    "living_room": 4,
    "kitchen": 5,
    "dining_room": 6,
}

softmax = nn.Softmax(dim=1)


@app.route("/", methods=["POST"])
def classify_room():
    """Receive everything in json!!!"""
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"decompressing image ...")
    image = data["image"]
    image = io.BytesIO(image)

    app.logger.debug(f"Reading a PIL image ...")
    image = Image.open(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    app.logger.debug(f"cropping and resizing image ...")
    image = crop_center_square(image)
    image = image.resize(size=(image_size, image_size))
    image = transform(image=np.array(image))["image"]
    image = image.to(device)

    app.logger.debug("Running the room-classifier ...")
    pred = net(image.unsqueeze(0))
    pred = softmax(pred).detach().cpu().numpy().squeeze()
    pred = pred.tolist()
    assert len(pred) == len(label2idx)
    results = {label: pred[idx] for label, idx in label2idx.items()}
    app.logger.info(f"results: {results}")

    response = jsonpickle.encode(results)
    app.logger.info("json-pickle is done.")

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10005)
