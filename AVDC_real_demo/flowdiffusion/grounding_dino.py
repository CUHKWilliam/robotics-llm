import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import os
from transformers import pipeline
import cv2
from PIL import Image

import torch
import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torchvision
from tqdm import tqdm
import pickle



# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8



def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

class GroundingDinoWrapper():
    def __init__(self):
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = "../../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
        GROUNDING_DINO_CHECKPOINT_PATH = "../../Grounded-Segment-Anything/groundingdino_swinb_cogcoor.pth"

        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_b"
        SAM_CHECKPOINT_PATH = "../../Grounded-Segment-Anything/sam_vit_b_01ec64.pth"

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.cuda('cuda:0')
        self.sam_predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

                
    def segment_with_text(self, image: np.ndarray, text) -> np.ndarray:
        image_original_size = image.shape
        image = cv2.resize(image, (256, 256))
        masks = []
        with torch.no_grad():
            # detect objects
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=[text],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{[text][class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                NMS_THRESHOLD
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            mask = segment(
                sam_predictor=self.sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            masks.append(mask)
        masks = np.concatenate(masks, axis=0).any(0)
        masks = cv2.resize(masks.astype(np.uint8), (image_original_size[1], image_original_size[0])).astype(np.bool_)
        return masks
