import time
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.utils import draw_bounding_boxes

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.9)
model.eval()
model = model.to(device)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
preprocess = preprocess.to(device)

while True:
    initial_time = time.time()
    img = torch.rand([3, 1080, 1920]).to(torch.uint8).to(device)
    batch = preprocess(img).unsqueeze(0)

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
    print(f"{round( 1 / (time.time() - initial_time), 1)}")
