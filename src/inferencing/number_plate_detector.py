import re
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
from transformers import PreTrainedModel, TrOCRProcessor, VisionEncoderDecoderModel

from inferencing.model import MultiNet


class NumberPlateDetector:
    def __init__(
        self,
        text_processor: TrOCRProcessor,
        text_model: VisionEncoderDecoderModel | PreTrainedModel,
        detection_model: MultiNet,
        detection_processor: transforms.Normalize,
    ):
        self.text_processor = text_processor
        self.text_model = text_model
        self.detection_model = detection_model
        self.detection_processor = detection_processor

    def obtain_text(
        self,
        single_image: torch.Tensor,
        device: str,
        debug: bool = False,
    ):
        with torch.no_grad():
            resized_image = resize(single_image, [256, 256])
            processed_image = self.detection_processor(resized_image)
            output_mask = self.detection_model(processed_image.unsqueeze(0)).squeeze(0)

        character_mask = (
            (output_mask.softmax(0)[0] > 0.5).cpu().numpy().astype(np.uint8)
        )
        # character_mask = cv2.morphologyEx(
        #     character_mask,
        #     cv2.MORPH_CLOSE,
        #     np.ones([5, 5]),
        #     iterations=5,
        # )
        # character_mask = cv2.morphologyEx(
        #     character_mask,
        #     cv2.MORPH_DILATE,
        #     np.ones([5, 5]),
        #     iterations=2,
        # )

        all_patch_images, all_patch_coodinates = self.generate_bounding_box(
            character_mask,
            single_image,
            debug,
        )

        if len(all_patch_images) == 0:
            return ""

        pixel_values = self.text_processor(
            all_patch_images,
            return_tensors="pt",
        ).pixel_values.to(device)

        generated_ids = self.text_model.generate(
            pixel_values,
            # max_length=20,
            # num_beams=2,
            # early_stopping=True,
            # no_repeat_ngram_size=3,
            # length_penalty=2,
        )

        generated_text = self.text_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        prefix_character = None
        number_character = None
        suffix_character = None

        # Find prefix
        for idx, generated_text in enumerate(generated_text):
            generated_text = generated_text
            generated_text = generated_text.lower()
            generated_text = re.sub("[^a-z0-9]", "", generated_text)

            x1, y1, x2, y2 = all_patch_coodinates[idx]
            if self.area(all_patch_coodinates[idx]) < 700:
                continue
            if ((y2 - y1) / (x2 - x1)) > 1:
                continue
            if re.match(r"^[a-z]{1,3}$", generated_text):
                prefix_character = generated_text
            elif re.match(r"^[a-z]$", generated_text):
                suffix_character = generated_text
            elif re.match(r"^[0-9]{1,4}$", generated_text):
                number_character = generated_text
            elif re.match(r"^([a-z]{1,4})?([0-9]{1,4})([a-z]{1,4})?$", generated_text):
                number_character = generated_text
                return number_character

        if debug:
            print(prefix_character, number_character, suffix_character)

        if number_character is not None and prefix_character is not None:
            parsed_license_plate = ""
            for item in [prefix_character, number_character, suffix_character]:
                if item is None:
                    continue
                parsed_license_plate += item
            return parsed_license_plate

        # Give a final pass
        # To find double row number plate
        # all_patch_images, all_patch_coodinates = self.sorted_highest_area(all_patch_images, all_patch_coodinates)

        return ""

    @staticmethod
    def sorted_highest_area(
        all_patch_images: List[Any], all_patch_coodinates: List[Any]
    ):
        unsorted_pair = []
        for i in range(len(all_patch_coodinates)):
            area = NumberPlateDetector.area(all_patch_coodinates[i])
            unsorted_pair.append((area, i))
        print(unsorted_pair)
        sorted_index = sorted(unsorted_pair, key=lambda x: x[0])
        return [all_patch_images[i] for area, i in sorted_index], [
            all_patch_coodinates[i] for area, i in sorted_index
        ]

    @staticmethod
    def generate_bounding_box(
        mask: np.ndarray,
        image: torch.Tensor,
        debug: bool = False,
    ) -> Tuple[List[Any], List[Any]]:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        _, image_height, image_width = image.size()
        mask_height, mask_width = mask.shape

        all_patch_images = []
        all_patch_coor = []

        for i in range(len(contours[0])):
            x, y, width, height = cv2.boundingRect(contours[0][i])

            # Rescale coordinate to original
            new_x = int(x * image_width / mask_width)
            new_y = int(y * image_height / mask_height)
            new_x_2 = new_x + int(width * image_width / mask_width)
            new_y_2 = new_y + int(height * image_height / mask_height)

            # new_x -= 1
            new_x_2 += 15

            all_patch_images.append(
                image[..., new_y:new_y_2, new_x:new_x_2] * 255,
            )
            all_patch_coor.append([new_x, new_y, new_x_2, new_y_2])

        return all_patch_images, all_patch_coor

    @staticmethod
    def midpoint(coor):
        return (
            (coor[2] - coor[0] / 2),
            (coor[3] - coor[1] / 2),
        )

    @staticmethod
    def area(coor):
        return (coor[2] - coor[0]) * (coor[3] - coor[1])

    @staticmethod
    def distance(coor_a, coor_b):
        return abs(coor_b[0] - coor_a[0]) + abs(coor_b[1] - coor_a[1])
