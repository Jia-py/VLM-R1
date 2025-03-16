# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math


from datasets import load_dataset, load_from_disk
from geopy.distance import geodesic

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation_image(example):
    QUESTION_TEMPLATE = '''
    Suppose you are an expert in geo-localization. Given an image, you are able to select the precise GPS coordinates where the image was taken from the GPS candidates. 
    For your reference, these are informations of some similar images (laitude and longitude of GPS candidates and their city, state, and country): {ref}, and these are informations of some dissimilar images: {ref_neg}. 
    Please first think step by step, then select the best prediction from the GPS candidates. 
    First, describe your step by step thought processes in <think>put your thoughts here</think> tags. Then, provide your final selection in the format of <answer></answer> tags (i.e., <answer>latitude, longitude</answer>)'''
    ref_gps = example["ref_gps"][:10]
    ref_texts = example["ref_texts"][:10]
    ref_texts = [[texts[1], texts[3], texts[5]] for texts in ref_texts]
    ref_gps_texts_lis = [f"({gps[0]}, {gps[1]}, {text})" for gps, text in zip(ref_gps, ref_texts)]
    random.shuffle(ref_gps_texts_lis)
    ref_gps_texts = '; '.join(ref_gps_texts_lis)
    ref_neg_gps = example["ref_gps_neg"][:10]
    ref_neg_texts = example["ref_texts_neg"][:10]
    ref_neg_texts = [[texts[1], texts[3], texts[5]] for texts in ref_neg_texts]
    ref_neg_texts_lis = [f"({gps[0]}, {gps[1]}, {text})" for gps, text in zip(ref_neg_gps, ref_neg_texts)]
    ref_neg_gps_texts = '; '.join(ref_neg_texts_lis)
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(ref=ref_gps_texts, ref_neg=ref_neg_gps_texts)},
                ],
            },
        ],
    }

def extract_location_answer(output_str):
    number_pattern = r'[-+]?\d+\.\d+'  # 匹配浮动数的正则表达式
    numbers = re.findall(number_pattern, output_str)
    numbers = numbers[-2:]
    
    if len(numbers) == 2:  # 确保找到了两个浮动数
        latitude = float(numbers[0])
        longitude = float(numbers[1])
        if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
            return 0.0, 0.0
        return latitude, longitude
    
    return 0.0, 0.0

def accuracy_reward(completions, solution, ref_gps, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    ref_gps = [ref[:10] for ref in ref_gps]
    for idx, (content, sol, ref) in enumerate(zip(contents, solution, ref_gps)):
        print('content:', content)
        reward = 0.0
        sol_match = re.findall(answer_tag_pattern, sol)
        latitude_sol, longitude_sol = extract_location_answer(sol_match[-1].strip())

        content_match = re.findall(r'<answer>(.*?)</answer>', content)
        if len(content_match) == 0:
            rewards.append(0)
            continue
        latitude_content, longitude_content = extract_location_answer(content_match[-1].strip())
        # print('latitude content, longitude content:', latitude_content, longitude_content)

        # check whether the output is in candidates
        flag = 0
        for lat, lon in ref:
            # print('refs:', lat, lon)
            # print('type:', type(lat))
            if latitude_content == lat and longitude_content == lon:
                flag = 1
                # print('match')
            else:
                # print('nomatch')
                pass
        
        if flag == 0: 
            rewards.append(0)
        else:
            distance_lis = []
            for lat, lon in ref:
                distance_lis.append(geodesic((latitude_sol, longitude_sol), (lat, lon)).km)

            content_distance = geodesic((latitude_sol, longitude_sol), (latitude_content, longitude_content)).km
            distance_lis.sort()
            reward = 0
            for i, dist in enumerate(distance_lis):
                if content_distance <= dist:
                    reward = len(ref) - i  # 距离越小，rank 越高，最大为 len(ref)
                    break
            
            rewards.append(reward)
        
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = load_dataset('Jia-py/mp16pro-rl')

    dataset = dataset.map(make_conversation_image)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        use_qlora=script_args.use_qlora,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.use_qlora = True
    main(script_args, training_args, model_args)
