import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from lisa.model.LISA import LISAForCausalLM
from lisa.model.llava import conversation as conversation_lib
from lisa.model.llava.mm_utils import tokenizer_image_token
from lisa.model.segment_anything.utils.transforms import ResizeLongestSide
from lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from peft import LoraConfig, get_peft_model

def parse_args():
    args = argparse.Namespace()
    args.version = "xinlai/LISA-7B-v1"
    args.precision = 'bf16'
    args.image_size = 1024
    args.model_max_length = 512
    args.lora_r = 0
    args.vision_tower="openai/clip-vit-large-patch14"
    args.local_rank = 0
    args.load_in_8bit = False
    args.load_in_4bit = False
    args.use_mm_start_end = True
    args.conv_type = "llava_v1"
    args.lora_alpha = 16
    args.lora_dropout = 0.05
    args.lora_target_modules = "q_proj,v_proj"
    return args


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


model = None
clip_image_processor = None
def main(image_np, prompt):
    args = parse_args()
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    transform = ResizeLongestSide(args.image_size)

    global model, clip_image_processor
    if model is None:
        model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
        lora_r = 0
        if lora_r > 0:
            def find_linear_layers(model, lora_target_modules):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if (
                        isinstance(module, cls)
                        and all(
                            [
                                x not in name
                                for x in [
                                    "visual_model",
                                    "vision_tower",
                                    "mm_projector",
                                    "text_hidden_fcs",
                                ]
                            ]
                        )
                        and any([x in name for x in lora_target_modules])
                    ):
                        lora_module_names.add(name)
                return sorted(list(lora_module_names))

            lora_alpha = args.lora_alpha
            lora_dropout = args.lora_dropout
            lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
            )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # model = get_peft_model(model, lora_config)
            # model.print_trainable_parameters()

        # model.resize_token_embeddings(len(tokenizer))
        if args.precision == "bf16":
            model = model.bfloat16().cuda()
        elif (
            args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
        ):
            vision_tower = model.get_model().get_vision_tower()
            model.model.vision_tower = None
            model = model.cuda().half()        
            model.model.vision_tower = vision_tower.half().cuda()
        elif args.precision == "fp32":
            model = model.float().cuda()
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(device=args.local_rank)

        clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

        ckpt_path = "../../AVDC/flowdiffusion/runs/lisa/model_1.pth"
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    pred_masks = torch.stack(pred_masks, dim=0)
    # THRESH = ((pred_masks.max()) - 1.).item()
    THRESH = 2.
    # ## TODO: for vis
    for i, pred_mask in enumerate(pred_masks):
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > THRESH
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug2.png", save_img)
        # import ipdb;ipdb.set_trace()
    # ## TODO: end vis
    pred_mask = torch.any(pred_masks > THRESH, dim=0)[0]
    return pred_mask.cpu()

def predict(image, prompt):
    pred_mask = main(image, prompt)
    return pred_mask

if __name__ == "__main__":
    image_path = 'door-close.png'
    prompt = 'Where should I grasp if I need to conduct task door open ? Please output segmentation mask.'
    main(cv2.imread(image_path), prompt)
