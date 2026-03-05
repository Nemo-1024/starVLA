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

    def _build_prompt_segments(self, instruction: str) -> tuple[str, str]:
        act_placeholder_block = " ".join([self.placeholder_token] * self.act_queries)
        flow_placeholder_block = " ".join([self.placeholder_token] * self.flow_queries)
        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
            base_prompt = cot_prompt.replace("{instruction}", instruction)
        else:
            base_prompt = instruction
        text_after_act = f"{base_prompt}\n{act_placeholder_block}" if act_placeholder_block else str(base_prompt)
        return text_after_act, flow_placeholder_block

    def build_qwenvl_inputs(
        self,
        *,
        images: Sequence[Sequence[Image.Image]],
        wrist_images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same length.")
        if len(wrist_images) != len(instructions):
            raise ValueError("Wrist images and instructions must have the same length.")

        messages = []
        for imgs, wrist_imgs, instruction in zip(images, wrist_images, instructions):
            # Keep primary views before ACT placeholders so ACT query cannot attend wrist images.
            content = [{"type": "image", "image": img} for img in imgs]
            text_after_act, flow_placeholder_block = self._build_prompt_segments(instruction)
            content.append({"type": "text", "text": text_after_act})
            # Insert wrist views between ACT and FLOW placeholders so only FLOW query can leverage them.
            content.extend({"type": "image", "image": img} for img in wrist_imgs)
            if flow_placeholder_block:
                content.append({"type": "text", "text": flow_placeholder_block})
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
