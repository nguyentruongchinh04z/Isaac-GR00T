# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np

import torch
from gr00t_inference import compare_predictions
from trt_model_forward import setup_tensorrt_engines
from trt_runner import GR00TN1d5TRTPolicy

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy


def validate_features(feat_name: str) -> None:
    trt_runner_feats = torch.load(ROOT / "debug" / "trt_runner" / f"{feat_name}.pt", weights_only=True)
    trt_gr00t_feats = torch.load(ROOT / "debug" / "trt_gr00t" / f"{feat_name}.pt", weights_only=True)

    if not torch.allclose(trt_runner_feats.cpu(), trt_gr00t_feats.cpu(), atol=1e-5):
        print(
            f"Warning: The features {feat_name} from TensorRT runner do not match the expected values from GR00T. "\
            "This may lead to different action predictions compared to PyTorch inference."
        )
        errors = (trt_runner_feats.cpu() - trt_gr00t_feats.cpu()).abs()
        print(f"Max absolute difference: {errors.max().item()}")
        print(f"Mean absolute difference: {errors.mean().item()}")
        input("Press Enter to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument(
        "--model-path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--trt-engine-path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["decord", "torchcodec"],
        help="Video backend to use for loading videos",
        default="decord",
    )
    parser.add_argument(
        "--vit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="ViT model dtype (fp16, fp8)",
        default="fp8",
    )
    parser.add_argument(
        "--llm-dtype",
        type=str,
        choices=["fp16", "nvfp4", "fp8"],
        help="LLM model dtype (fp16, nvfp4, fp8)",
        default="nvfp4",
    )
    parser.add_argument(
        "--dit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="DiT model dtype (fp16, fp8)",
        default="fp8",
    )
    args = parser.parse_args()

    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data", "robot_sim.PickNPlace")
    EMBODIMENT_TAG = "gr1"
    denoising_steps = 4
    device = "cuda"

    # Load data config
    from gr00t.experiment.data_config import load_data_config
    data_config = load_data_config("fourier_gr1_arms_only")
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Load dataset and extract the first step data
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )
    print("\n\n\n========== DATASET LOADED ==========")
    
    step_data = dataset[0]
    print("\n\n\n========== STEP DATA SCHEMA ==========")
    for key, value in step_data.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)
        
    # Set environment variable to enable debug mode in TensorRT forward pass
    print("\n\n\n========== RUNNING NATIVE PYTORCH INFERENCE ==========")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    if not hasattr(policy.model.action_head, "init_actions"):
        policy.model.action_head.init_actions = torch.zeros(
            1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim,
            dtype=torch.float16,
            device=device,
        )
    pred_pytorch = policy.get_action(step_data)

    # Setup TensorRT engines with debug mode enabled for getting debug features
    print("\n\n\n========== SETTING UP TENSORRT ENGINES WITH DEBUG MODE ENABLED ==========")
    os.environ["TENSORRT_FORWARD_DEBUG"] = "1" 
    setup_tensorrt_engines(
        policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
    )
    policy.get_action(step_data)
    
    # Free up GPU memory before loading the TensorRT Runner
    init_actions = policy.model.action_head.init_actions
    del policy
    torch.cuda.empty_cache()

    # Load the TRT Runner debug outputs and compare with PyTorch outputs
    print("\n\n\n========== RUNNING TENSORRT RUNNER ==========")
    policy = GR00TN1d5TRTPolicy(
        model_path=args.model_path,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        trt_engine_path=args.trt_engine_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        denoising_steps=denoising_steps,
        device=device,
        debug_pipeline=True
    )
    policy.init_actions = init_actions
    pred_tensorrt = policy.get_action(step_data)

    
    # Compare predictions
    print("\n\n\n========== VALIDATING INTERMEDIATE FEATURES ==========")
    validate_features("backbone_embs")
    validate_features("vl_embs")
    validate_features("state_features")
    for t in range(denoising_steps):
        validate_features(f"action_features_before_pos_embed_t{t}")
        validate_features(f"action_features_t{t}")
        validate_features(f"sa_embs_t{t}")
        validate_features(f"model_output_t{t}")
        validate_features(f"pred_velocity_t{t}")
    print("*** Validation completed. All features were matched. ***")
    
    print("\n\n\n========== COMPARING TENSORRT AND PYTORCH PREDICTIONS ==========")
    compare_predictions(pred_tensorrt, pred_pytorch)
