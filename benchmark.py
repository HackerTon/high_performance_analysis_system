import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.utils import draw_bounding_boxes

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.9)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

cam = cv2.VideoCapture(0)
while True:
    initial_time = time.time()
    check, frame = cam.read()

    img = torch.permute(torch.Tensor(frame[:, :, [2, 1, 0]]), [2, 0, 1]).to(torch.uint8)
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    with torch.no_grad():
        prediction = model(batch)[0]
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(
            img,
            boxes=prediction["boxes"],
            labels=labels,
            colors="red",
            width=4,
            font_size=30,
        )

    # cv2.imshow('video', torch.permute(box, [1, 2, 0]).numpy()[..., [2, 1, 0]])

    print(f"{round( 1 / (time.time() - initial_time), 1)}")

    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
