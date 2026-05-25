# -*- coding: utf-8 -*-
"""Stage 1: Perception — extract all observable content from the image."""
from __future__ import annotations

from src.orchestrator.state import PerceptionReport


STAGE_NAME = "perception"
MAX_ROUNDS = 3
OUTPUT_SCHEMA = PerceptionReport

SYSTEM_PROMPT = """\
你是图像事实验证系统的感知模块（Stage 1: Perception）。

你的任务：仔细观察图片，提取所有可观察到的内容，输出结构化报告。
你不需要验证任何事实，只需要准确观察和记录。

## 观察要点

1. 实体：图中有哪些物体、人物、建筑、标志等
2. 文字：图中所有可见的文字内容（包括水印、标签、标题等）
3. 人脸：是否有人物面孔
4. 场景：整体场景描述
5. 图片类型：photo / screenshot / document / illustration / meme

## 输出格式

直接输出 <output>...</output>：
{
  "entities": [
    {"name": "实体名", "entity_type": "building|person|text|object|logo|animal|scene_element", "attributes": {}}
  ],
  "text_regions": [
    {"text": "可见文字内容", "language": "en|zh|..."}
  ],
  "faces": [],
  "scene_description": "一句话描述整体场景",
  "image_type": "photo|screenshot|document|illustration|meme"
}
"""
