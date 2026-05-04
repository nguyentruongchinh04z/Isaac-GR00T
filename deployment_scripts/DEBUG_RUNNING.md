# TensorRT Runner Testing
## Preparation
- GR00T-N1.5-3B weights
- GR00T Tensorrt Engines for components
## Run Validating
```bash
python deployment_scripts/gr00t_debug.py --model-path weights/GR00T-N1.5-3B/ --trt-engine-path gr00t_engine/ --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16
```