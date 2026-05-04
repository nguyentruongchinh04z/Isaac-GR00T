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

import numpy as np

import torch
from trt_model_forward import setup_tensorrt_engines
from trt_runner import GR00TN1d5TRTPolicy

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


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
        denoising_steps=4,
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
        denoising_steps=4,
        device=device,
    )
    policy.init_actions = init_actions
    pred_tensorrt = policy.get_action(step_data)


    # Compare predictions
    print("\n\n\n========== COMPARING TENSORRT AND PYTORCH PREDICTIONS ==========")
    compare_predictions(pred_tensorrt, pred_pytorch)
