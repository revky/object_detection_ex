import json
import os
import gdown
import torch
from PIL import Image
from detecto.core import Model

from src.custom_detecto import save_labeled_image

URL = "https://drive.google.com/drive/folders/1ImIDSw_GMuBbyJl7lBq51nGSR-cFy7Mk?usp=drive_link"
DATA_DIR = "data"
MODEL_PATH = "models/model.pth"
CLASSES = ["maseczka"]
OUTPUTS_PATH = "outputs"


def load_data(url: str, output: str):
    files = []
    try:
        print(f"Started loading files from {url}")
        files = gdown.download_folder(url, remaining_ok=True, quiet=True, output=output)
    except Exception as e:
        print(f"Following exception occurred", e)
    finally:
        print(f"Loaded {len(files)} files")


def filter_predictions(labels, boxes, scores, threshold=0.6):
    filter_mask = scores >= threshold
    filtered_boxes = boxes[filter_mask]
    filtered_scores = scores[filter_mask]

    indices = torch.where(filter_mask)[0]
    filtered_labels = [labels[i] for i in indices]

    return filtered_labels, filtered_boxes, filtered_scores


def save_predictions(file_name, labels, boxes, scores):
    predictions_list = []
    for label, box, score in zip(labels, boxes, scores):
        pred = {
            "label": label,
            "bbox": box.numpy().tolist(),
            "score": score.numpy().tolist()
        }
        predictions_list.append(pred)

    file_name = file_name + ".json"
    with open(file_name, "w") as f:
        json.dump(predictions_list, f)


if __name__ == '__main__':
    if len(os.listdir(DATA_DIR)) == 0:
        print(f"Fetching data to {DATA_DIR}")
        load_data(URL, DATA_DIR)
    else:
        print(f"Found data in data directory")

    try:
        model = Model.load(MODEL_PATH, CLASSES)
        print("Model loaded successfully")
    except Exception:
        print("Error loading the model, exiting")
        exit()

    while True:
        try:
            img_uri = input("Provide path to image:")
            img = Image.open(img_uri)
            print("Image loaded successfully")
            break
        except Exception:
            print("Error loading image")

    while True:
        threshold = input("Provide prediction threshold:")
        print("Making predictions")
        predictions = model.predict(img)
        labels, boxes, scores = filter_predictions(*predictions, threshold=float(threshold))
        valid_predictions = len(scores)

        if valid_predictions == 0:
            print("Couldn't find any predictions for given threshold")
        else:
            print(f"Found {valid_predictions}, saving to /outputs")
            outputs_file_name = f"{OUTPUTS_PATH}/{img_uri.split('/')[-1].split('.')[0]}_predictions"

            save_labeled_image(outputs_file_name, img, labels, boxes)
            save_predictions(outputs_file_name, labels, boxes, scores)
            break
