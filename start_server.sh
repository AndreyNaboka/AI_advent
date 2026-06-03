#!/bin/bash
cd ~/code/AI_advent  
source venv/bin/activate
python3 -m llama_cpp.server --model ~/code/models/qwen3-4b/qwen3-4b-instruct-2507-q8_0.gguf --n_gpu_layers 99 --port 8080 --n_ctx 8192
