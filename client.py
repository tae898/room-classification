"""
This is just a simple client example. Hack it as much as you want. 
"""
import argparse
import io
import json
import logging

import jsonpickle
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def send_to_servers(binary_image, url_room: str) -> dict:
    """Send a binary image to the two servers.

    Args
    ----
    binary_image: binary image
    url_room: url of the room-classifier server

    Returns
    -------
    room_classification results

    """
    data = {"image": binary_image}

    logging.debug(f"sending image to server...")
    data = jsonpickle.encode(data)
    response = requests.post(url_room, json=data)
    logging.info(f"got {response} from server!...")
    response = jsonpickle.decode(response.text)

    logging.info(f"room-classification results: {response}...")

    return response


def annotate_image(image: Image.Image, room_classification: dict) -> None:
    """Annotate a given image. This is done in-place. Nothing is returned.

    Args
    ----
    image: Pillow image
    room_classification:
    """
    logging.debug(f"annotating image ...")

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)

    label = max(room_classification, key=room_classification.get)
    prob = round(room_classification[label], 4)

    width, height = image.size
    center = width // 2, height // 2
    draw.text(
        center,
        f"{label}: {str(round(prob * 100))}" + str("%"),
        fill=(255, 0, 0),
        font=font,
    )


def run_image(url_room: str, image_path: str, save: bool = True):
    """Run age-gender on the image.

    Args
    ----
    url_room: url of the room-classifier server
    image_path: image path
    save: whether to save in disk or not.

    """
    logging.debug(f"loading image ...")
    with open(image_path, "rb") as stream:
        binary_image = stream.read()

    room_classification = send_to_servers(binary_image, url_room)

    if save:
        results_path = image_path + ".json"
        with open(results_path, "w") as stream:
            json.dump(room_classification, stream, indent=4)


def annotate_fps(image: Image.Image, fps: int) -> None:
    """Annotate fps on a given image.

    Args
    ----
    image: Pillow image
    fps: frames per second

    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)
    draw.text((0, 0), f"FPS: {fps} (Press q  to exit.)", fill=(0, 0, 255), font=font)


def run_webcam(url_room: str, camera_id: int):

    import time

    import cv2

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        start_time = time.time()  # start time of the loop
        # Capture frame-by-frame
        ret, image_BGR = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

        image_PIL = Image.fromarray(image_RGB)
        binary_image = io.BytesIO()
        image_PIL.save(binary_image, format="JPEG")
        binary_image = binary_image.getvalue()

        room_classification = send_to_servers(binary_image, url_room)

        annotate_image(image_PIL, room_classification)

        fps = int(1.0 / (time.time() - start_time))

        annotate_fps(image_PIL, fps)

        cv2.imshow("frame", cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify room type")
    parser.add_argument("--url-room", type=str, default="http://127.0.0.1:10005/")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-id", type=int, default="0", help="ffplay /dev/video0")
    parser.add_argument("--mode", type=str, default="image", help="image or webcam")

    args = vars(parser.parse_args())

    logging.info(f"arguments given to {__file__}: {args}")

    mode = args.pop("mode")
    if mode == "image":
        assert args["image_path"] is not None
        del args["camera_id"]
        run_image(**args)
    else:
        del args["image_path"]
        run_webcam(**args)
