from typing import Tuple

import os
import tree
import torch
import atexit
import ctypes
import tensorrt as trt
import torch.nn as nn

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

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


class GR00TN1d5FakePolicy(Qwen3ForCausalLM):
    def __init__(self, config=None):
        super().__init__(config)
        self.num_patches = 196
        self.use_pixel_shuffle = False
        self.image_token_index = 151669
        self.num_inference_timesteps = 4
        self.num_timestep_buckets = 1000
        self.action_horizon = 16
        self.device = "cuda"
        self.input_embedding_dim = 1536
        self.add_pos_embed = True
        self.max_seq_len = 1024
        self.num_target_vision_tokens = 32
        self.vit_hidden_size = 1152
        self.llm_hidden_size = 2048
        
        self.action_dim = None

        self.future_tokens = nn.Embedding(self.num_target_vision_tokens, self.input_embedding_dim)
        self.embedding_layer = self.get_input_embeddings()
        self.position_embedding = nn.Embedding(self.max_seq_len, self.input_embedding_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.vit_hidden_size, self.llm_hidden_size),
        )

    def load(self):
        raise NotImplementedError


class GR00TN1d5ModelRunner(GR00TN1d5FakePolicy):
    def __init__(self,trt_engine_path, vit_dtype="fp16", llm_dtype="fp16", dit_dtype="fp16"):
        super().__init__()
        self.vit_engine = Engine(
            os.path.join(trt_engine_path, f"vit_{vit_dtype}.engine")
        )
        self.llm_engine = Engine(
            os.path.join(trt_engine_path, f"llm_{llm_dtype}.engine")
        )
        self.vlln_vl_self_attention_engine = Engine(
            os.path.join(trt_engine_path, "vlln_vl_self_attention.engine")
        )
        self.action_encoder_engine = Engine(
            os.path.join(trt_engine_path, "action_encoder.engine")
        )
        self.action_decoder_engine = Engine(
            os.path.join(trt_engine_path, "action_decoder.engine")
        )
        self.DiT_engine = Engine(
            os.path.join(trt_engine_path, f"DiT_{dit_dtype}.engine")
        )
        self.state_encoder_engine = Engine(
            os.path.join(trt_engine_path, "state_encoder.engine")
        )

    def set_frozen_modules_to_eval_mode(self):
        self.future_tokens.eval()
        self.embedding_layer.eval()
        self.position_embedding.eval()
        self.mlp1.eval()

    def eagle_tensorrt_forward(self, vl_input):
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        vl_input = eagle_input

        self.set_frozen_modules_to_eval_mode()
        batch_size = vl_input["pixel_values"].shape[0]
        position_ids = torch.arange(self.num_patches, device="cuda").expand((batch_size, -1))
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

        return BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": vl_input["attention_mask"],
            }
        )

    def action_head_tensorrt_forward(self, backbone_output, action_input):
        # backbone_output = self.process_backbone_output(backbone_output)
        if backbone_output.backbone_features.dtype != torch.float16:
            backbone_output.backbone_features = backbone_output.backbone_features.to(torch.float16)
        self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
            "backbone_features", backbone_output.backbone_features.shape
        )
        backbone_output.backbone_features = self.vlln_vl_self_attention_engine(
            backbone_output.backbone_features
        )["output"]
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        batch_size = vl_embs.shape[0]

        if action_input.state.dtype != torch.float16:
            action_input.state = action_input.state.to(torch.float16)

        if embodiment_id.dtype != torch.int64:
            embodiment_id = embodiment_id.to(torch.int64)

        if vl_embs.dtype != torch.float16:
            vl_embs = vl_embs.to(torch.float16)

        # Embed state with batch processing

        self.state_encoder_engine.set_runtime_tensor_shape("state", action_input.state.shape)
        self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        state_features = self.state_encoder_engine(action_input.state, embodiment_id)["output"]

        # Set initial actions as the sampled noise.
        device = vl_embs.device

        # This attribute is used to ensure the same actions is used for both PyTorch and TensorRT inference
        if hasattr(self, "init_actions"):
            actions = self.init_actions.expand((batch_size, -1, -1))
        else:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory with batch processing
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

            self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
            self.action_encoder_engine.set_runtime_tensor_shape(
                "timesteps_tensor", timesteps_tensor.shape
            )
            self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
            action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
                "output"
            ]

            # Maybe add position embedding.
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(
                torch.float16
            )
            # Run model forward with batch processing
            if vl_embs.dtype != torch.float16:
                vl_embs = vl_embs.to(torch.float16)

            self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
            self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
            model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]

            self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
            self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
            pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = BatchFeature(data=inputs)
        action_inputs = BatchFeature(data=inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.eagle_tensorrt_forward(backbone_inputs)
        action_head_outputs = self.action_head_tensorrt_forward(backbone_outputs, action_inputs)
        return action_head_outputs


if __name__ == "__main__":
    trt_engine_path = "/home/develop/VR_VLA-OPT_ws/Isaac-GR00T/gr00t_engine"
    vit_dtype = "fp16"
    llm_dtype = "fp16"
    dit_dtype = "fp16"

    vit_engine = Engine(
        os.path.join(trt_engine_path, f"vit_{vit_dtype}.engine")
    )
    llm_engine = Engine(
        os.path.join(trt_engine_path, f"llm_{llm_dtype}.engine")
    )
    vlln_vl_self_attention_engine = Engine(
        os.path.join(trt_engine_path, "vlln_vl_self_attention.engine")
    )
    action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )
    DiT_engine = Engine(
        os.path.join(trt_engine_path, f"DiT_{dit_dtype}.engine")
    )
    state_encoder_engine = Engine(
        os.path.join(trt_engine_path, "state_encoder.engine")
    )