from typing import Any, Optional, Tuple, Dict, Union

import os
import tree
import json
import torch
import atexit
import ctypes
import numpy as np
import torch.nn as nn
import tensorrt as trt
from pathlib import Path
from safetensors.torch import load_file as safe_load_file

ROOT = Path(__file__).resolve().parents[1]

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform


####################################################################################################
###################################### HELPER FUNCTIONS ############################################
####################################################################################################
def torch_type(trt_type):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve TensorRT datatype to an equivalent numpy datatype. {trt_type}"
    )


def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data


####################################################################################################
################################## TENSORRT ENGINE PART ############################################
####################################################################################################
class Engine(object):
    def __init__(self, file, plugins=[]):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert (
                self.handle is not None
            ), f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for iarg, x in enumerate(args):
            name, shape, dtype = self.in_meta[iarg]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor dtype, excepted dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, excepted device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert (
                runtime_shape == x.shape
            ), f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert (
                x.is_cuda
            ), f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for item in self.out_meta:
            name = item[0]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(
                *runtime_shape, dtype=item[2], device=reference_tensors[0].device
            )
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(
            self.out_meta
        ), f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"

        if return_list:
            return [
                reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)
            ]
        else:
            return {
                item[0]: reference_tensors[len(self.in_meta) + i]
                for i, item in enumerate(self.out_meta)
            }
        

#############################################################################################################
########################################## TENSORRT POLICY PART #############################################
#############################################################################################################
class GR00TN1d5TRTPolicy(nn.Module):
    def __init__(
        self, 
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        trt_engine_path: str,     
        vit_dtype: str = "fp16", 
        llm_dtype: str = "fp16", 
        dit_dtype: str = "fp16",
        device: Union[int, str] = "cuda",
        denoising_steps: Optional[int] = None,
        debug_pipeline: bool = False,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.trt_engine_path = Path(trt_engine_path)
        self.vit_dtype = vit_dtype
        self.llm_dtype = llm_dtype
        self.dit_dtype = dit_dtype
        self.debug_pipeline = debug_pipeline
        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.device = torch.device(device)
        assert self.device.type == "cuda", "TensorRT inference requires CUDA device"

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag
        self._is_loaded = False
        self.init_actions = None

        # Policy Configuration
        self.num_patches = 256
        self.use_pixel_shuffle = False
        self.vocab_size = 151680
        self.image_token_index = 151669
        self.num_inference_timesteps = denoising_steps or 4
        self.num_timestep_buckets = 1000
        self.action_horizon = 16
        self.backbone_embedding_dim = 2048
        self.input_embedding_dim = 1536
        self.add_pos_embed = True
        self.max_seq_len = 1024
        self.num_target_vision_tokens = 32
        self.llm_hidden_size = 2048
        self.vit_hidden_size = 1152
        self.action_dim = 32

        # Pytorch Modules
        self.embedding_layer = nn.Embedding(self.vocab_size, self.backbone_embedding_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.input_embedding_dim)
        self.future_tokens = nn.Embedding(self.num_target_vision_tokens, self.input_embedding_dim)
        self.mlp1 = nn.Sequential(nn.Linear(self.vit_hidden_size, self.llm_hidden_size))

        # TensorRT Engines
        self.vit_engine = None
        self.llm_engine = None
        self.vlln_vl_self_attention_engine = None
        self.action_encoder_engine = None
        self.action_decoder_engine = None
        self.DiT_engine = None
        self.state_encoder_engine = None

        # Load the Policy
        self.load()

    #####################################################################################################
    ######################################### INFERENCE METHODS #########################################
    #####################################################################################################
    def eagle_tensorrt_forward(self, vl_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        vl_input = eagle_input

        self.set_frozen_modules_to_eval_mode()
        batch_size = vl_input["pixel_values"].shape[0]
        position_ids = torch.arange(self.num_patches, device=self.device).expand((batch_size, -1))
        if vl_input["pixel_values"].dtype != torch.float16:
            vl_input["pixel_values"] = vl_input["pixel_values"].to(torch.float16)

        assert (
            vl_input["pixel_values"].shape[0] <= 8
        ), "Batch size must be <= 8 because TensorRT engine was built with max_batch_size=8, "
        "you can try to adjust the max_batch_size in the build_engine.sh script and rebuild the engine."

        self.vit_engine.set_runtime_tensor_shape("pixel_values", vl_input["pixel_values"].shape)
        self.vit_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
        vit_embeds = self.vit_engine(vl_input["pixel_values"], position_ids)["vit_embeds"]
        vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        # Get input_ids from vl_input and convert to embeddings
        input_ids = vl_input["input_ids"]
        input_embeds = self.embedding_layer(input_ids)

        # Convert to float16 if needed (TensorRT engine expects float16)
        if input_embeds.dtype != torch.float16:
            input_embeds = input_embeds.to(torch.float16)
        if vit_embeds.dtype != torch.float16:
            vit_embeds = vit_embeds.to(torch.float16)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids_flat = input_ids.reshape(B * N)
        selected = input_ids_flat == self.image_token_index
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        self.llm_engine.set_runtime_tensor_shape("inputs_embeds", input_embeds.shape)
        self.llm_engine.set_runtime_tensor_shape("attention_mask", vl_input["attention_mask"].shape)
        embeddings = self.llm_engine(input_embeds, vl_input["attention_mask"])["embeddings"]
        self.maybe_save_debug_features(embeddings, "backbone_embs")
        
        return {
            "backbone_features": embeddings,
            "backbone_attention_mask": vl_input["attention_mask"],
        }

    def action_head_tensorrt_forward(
        self, 
        backbone_output: dict[str, torch.Tensor], 
        action_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Run VL self-attention engine with batch processing
        if backbone_output["backbone_features"].dtype != torch.float16:
            backbone_output["backbone_features"] = backbone_output["backbone_features"].to(torch.float16)
        self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
            "backbone_features", backbone_output["backbone_features"].shape
        )
        backbone_output["backbone_features"] = self.vlln_vl_self_attention_engine(
            backbone_output["backbone_features"]
        )["output"]

        # Prepare inputs for action generation loop
        vl_embs = backbone_output["backbone_features"]
        if vl_embs.dtype != torch.float16:
            vl_embs = vl_embs.to(torch.float16)
        self.maybe_save_debug_features(vl_embs, "vl_embs")

        embodiment_id = action_input["embodiment_id"]
        batch_size = vl_embs.shape[0]

        if action_input["state"].dtype != torch.float16:
            action_input["state"] = action_input["state"].to(torch.float16)

        if embodiment_id.dtype != torch.int64:
            embodiment_id = embodiment_id.to(torch.int64)

        # Embed state with batch processing
        self.state_encoder_engine.set_runtime_tensor_shape("state", action_input["state"].shape)
        self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        state_features = self.state_encoder_engine(action_input["state"], embodiment_id)["output"]
        self.maybe_save_debug_features(state_features, "state_features")

        # Set initial actions as the sampled noise.
        # This attribute is used to ensure the same actions is used for both PyTorch and TensorRT inference
        if hasattr(self, "init_actions"):
            actions = self.init_actions.expand((batch_size, -1, -1))
        else:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=self.device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory with batch processing
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=self.device)

            self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
            self.action_encoder_engine.set_runtime_tensor_shape(
                "timesteps_tensor", timesteps_tensor.shape
            )
            self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
            action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
                "output"
            ]
            self.maybe_save_debug_features(action_features, f"action_features_before_pos_embed_t{t}")
            
            # Maybe add position embedding.
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
                action_features = action_features + pos_embs
            self.maybe_save_debug_features(action_features, f"action_features_t{t}")

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(
                torch.float16
            )
            self.maybe_save_debug_features(sa_embs, f"sa_embs_t{t}")

            # Run model forward with batch processing
            self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
            self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
            model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]
            self.maybe_save_debug_features(model_output, f"model_output_t{t}")

            self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
            self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
            pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
            pred_velocity = pred[:, -self.action_horizon :]
            self.maybe_save_debug_features(pred_velocity, f"pred_velocity_t{t}")

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return {"action_pred": actions}

    @torch.inference_mode()
    def infer(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.set_frozen_modules_to_eval_mode()
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.eagle_tensorrt_forward(backbone_inputs)
        action_head_outputs = self.action_head_tensorrt_forward(backbone_outputs, action_inputs)
        return action_head_outputs
    
    ########################################################################################################
    ###################################### GET ACTION METHODS ##############################################
    ########################################################################################################
    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform.apply(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform.unapply(action)
    

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)
        
        # Apply transforms and run inference
        normalized_input = self.apply_transforms(obs_copy)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized_action = self.infer(normalized_input)["action_pred"]
        unnormalized_action = self.unapply_transforms({"action": normalized_action.float().cpu()})

        # If the input is not batched, squeeze the batch dimension from the output action
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    ########################################################################################################
    ############################# HELPER METHODS FOR INFERENCE PREPARATION AND LOADING #####################
    ########################################################################################################
    def set_frozen_modules_to_eval_mode(self) -> None:
        self.future_tokens.eval()
        self.embedding_layer.eval()
        self.position_embedding.eval()
        self.mlp1.eval()
    
    def prepare_input(
        self, inputs: dict[str, torch.Tensor]
    ) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        backbone_inputs = inputs.copy()
        action_inputs = inputs.copy()

        def to_device_with_maybe_dtype(x: torch.Tensor) -> torch.Tensor:
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=torch.bfloat16)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs
    
    def load(self) -> None:
        if self._is_loaded:
            raise RuntimeError("The policy has already been loaded.")
        assert self.model_path.exists(), f"Model path {self.model_path} does not exist."
        assert self.trt_engine_path.exists(), f"TensorRT engine path {self.trt_engine_path} does not exist."
        
        # Load Pytorch State Dict from .safetensors files
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if len(safetensor_files) == 0:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_path}")
        key_mappings = {
            "mlp1": "backbone.eagle_model.mlp1", 
            "position_embedding": "action_head.position_embedding", 
            "future_tokens": "action_head.future_tokens", 
            "embedding_layer": "backbone.eagle_model.language_model.model.embed_tokens"
        }
        out_state_dict = {}
        for safetensor_file in safetensor_files:
            state_dict = safe_load_file(safetensor_file)
            for map_key, state_key in key_mappings.items():
                related_state_dict = {
                    key: val for key, val in state_dict.items() if key.__contains__(state_key)
                }
                if len(related_state_dict):
                    for key, val in related_state_dict.items():
                        out_key = key.replace(state_key, map_key)
                        out_state_dict[out_key] = val
        missing_keys = set(key_mappings.keys()) - set([k.split(".")[0] for k in out_state_dict.keys()])
        if len(missing_keys):
            raise KeyError(f"Missing keys in the loaded state dict: {missing_keys}")
        print("Loading Pytorch Modules done: ", self.load_state_dict(out_state_dict, assign=True))
        self.to(self.device)

        # Load TensorRT Engines
        self.vit_engine = Engine(str(self.trt_engine_path / f"vit_{self.vit_dtype}.engine"))
        self.llm_engine = Engine(str(self.trt_engine_path / f"llm_{self.llm_dtype}.engine"))
        self.vlln_vl_self_attention_engine = Engine(str(self.trt_engine_path / "vlln_vl_self_attention.engine"))
        self.action_encoder_engine = Engine(str(self.trt_engine_path / "action_encoder.engine"))
        self.action_decoder_engine = Engine(str(self.trt_engine_path / "action_decoder.engine"))
        self.DiT_engine = Engine(str(self.trt_engine_path / f"DiT_{self.dit_dtype}.engine"))
        self.state_encoder_engine = Engine(str(self.trt_engine_path / "state_encoder.engine"))
        
        self._load_metadata()
        self._load_horizons()
        self._is_loaded = True  


    def _load_metadata(self) -> None:
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = self.model_path / "experiment_cfg" / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self) -> None:
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray) -> None:
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def maybe_save_debug_features(self, features: torch.Tensor, feat_name: str) -> None:
        """Utility function to save intermediate features for debugging."""
        if not self.debug_pipeline:
            return
        save_file = ROOT / "debug" / "trt_runner" / f"{feat_name}.pt"
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(features.cpu(), save_file)


if __name__ == "__main__":
    TRT_ENGINE_PATH = ROOT / "gr00t_engine"
    MODEL_PATH = ROOT / "weights" / "GR00T-N1.5-3B"
    EMBODIMENT_TAG = "gr1"

    DATASET_PATH = ROOT / "demo_data" / "robot_sim.PickNPlace"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy
    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Create the policy
    policy = GR00TN1d5TRTPolicy(
        model_path=MODEL_PATH,
        trt_engine_path=TRT_ENGINE_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )


    # Get a step of data from the dataset and run inference
    init_actions = torch.zeros(
        1, policy.action_horizon, policy.action_dim,
        dtype=torch.float16,
        device=policy.device,
    )

    step_data = dataset[0]
    print("\n\n ====================================")
    for key, value in step_data.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)

    policy.init_actions = init_actions.clone()
    pred_tensorrt = policy.get_action(step_data)
    for key, value in pred_tensorrt.items():
        print(key, value.shape)

    
    del policy
    torch.cuda.empty_cache()

    # Load the original PyTorch model for comparison
    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    
    step_data = dataset[0]
    policy.model.action_head.init_actions = init_actions.clone()
    pred_torch = policy.get_action(step_data)
    for key, value in pred_torch.items():
        print(key, value.shape)

    del policy
    torch.cuda.empty_cache()
