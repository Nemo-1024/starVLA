from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image


class LatentWorldPolicyVLMAdapter(nn.Module):
    """Policy-side adapter for Qwen-VL prompt/processor encoding."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        config: Any,
        placeholder_token: str,
        act_queries: int,
        flow_queries: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self.placeholder_token = str(placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)

    def _build_prompt_with_query(self, instruction: str) -> str:
        placeholder_block = " ".join([self.placeholder_token] * (self.act_queries + self.flow_queries))
        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
            base_prompt = cot_prompt.replace("{instruction}", instruction)
        else:
            base_prompt = instruction
        return f"{base_prompt}\n{placeholder_block}"

    def build_qwenvl_inputs(
        self,
        *,
        images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same length.")

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = self._build_prompt_with_query(instruction)
            content.append({"type": "text", "text": prompt})
            messages.append([
                {
                    "role": "user",
                    "content": content,
                }
            ])

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
