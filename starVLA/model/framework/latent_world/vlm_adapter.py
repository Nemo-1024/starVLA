from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LatentWorldQwenVLInterface(nn.Module):
    """Adapter that aligns latent world input construction with other Qwen frameworks."""

    def __init__(self, *, model: nn.Module, processor: Any, config: Any) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        del kwargs
        if solutions is not None:
            raise ValueError("LatentWorldQwenVLInterface does not support `solutions`.")

        messages = []
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same length.")

        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            content.append({"type": "text", "text": prompt})
            messages.append([
                {
                    "role": "user",
                    "content": content,
                }
            ])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        return batch_inputs.to(model_device)
