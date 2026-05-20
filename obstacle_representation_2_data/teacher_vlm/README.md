# Obstacle Representation 2 Teacher Cache

This directory is reserved for cached offline VLM teacher outputs.

The first Scheme A+2 training run used existing OA-LLM `llm_strategy` metadata plus depth/pointcloud affordance rules, so it did not make new online VLM calls. Future runs can store per-frame VLM boxes, masks, or semantic notes here and merge them during dataset building.
