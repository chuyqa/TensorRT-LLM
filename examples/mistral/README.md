# Mistral

This document shows how to build and run a Mistral model in TensorRT-LLM on single GPU, single node multi-GPU, and multi-node multi-GPU.

## Overview

The TensorRT-LLM Mistral implementation can be found in [tensorrt_llm/models/mistral/model.py](../../tensorrt_llm/models/mistral/model.py). The TensorRT-LLM Mistral example code is located in [`examples/mistral`](./). There are three main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Mistral model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix
  * FP16
  * BF16
  * FP8
  * STRONGLY TYPED
  * FP8 KV CACHE
  * Tensor Parallel

## Usage

The TensorRT-LLM Mistral example code is located at [examples/mistral](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF Mistral checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/mistral.

```bash
# Setup git-lfs
git lfs install
# mistralai/Mistral-7B-v0.1
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-v0.1
# mistralai/Mistral-7B-Instruct-v0.1
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mistral-7B-Instruct-v0.1
 ```

TensorRT-LLM Mistral builds TensorRT engine(s) from HF checkpoint.
If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires a single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallel building to make the engine building process faster by adding `--parallel_build` argument.
Please note that currently `parallel_build` feature only supports single node.

Here are some examples:
```bash
# Build a single-GPU float16 engine from HF weights.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance
python build.py --model_dir mistralai/Mistral-7B-v0.1 \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir mistral/rw-1b/trt_engines/fp16/1-gpu/

# Single GPU on mistral-7b-instruct
# --use_gpt_attention_plugin is necessary for rotary positional embedding (RoPE)
python build.py --model_dir mistralai/Mistral-7B-v0.1 \
                --dtype bfloat16 \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir mistral/7b-v1/trt_engines/bf16/1-gpu/ \
                --world_size 1

# Use 2-way tensor parallelism on mistral-7b-instruct
python build.py --model_dir mistral/7b-instruct \
                --dtype bfloat16 \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir mistral/7b-v2/trt_engines/bf16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Use 2-way tensor parallelism and 2-way pipeline parallelism on mistral-7b-instruct
python build.py --model_dir mistralai/Mistral-7B-Instruct-v0.1 \
                --dtype bfloat16 \
                --use_gemm_plugin bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir mistralai/Mistral-7B-Instruct-v0.1/trt_engines/bf16/2-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

```

Note that in order to use N-way tensor parallelism, the number of attention heads must be a multiple of N.
For example, you can't configure 2-way tensor parallism for [mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) or [mistral-7b-instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), because the number of attention heads is 71 (not divisible by 2).


#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

Now quantize HF Mistral weights as follows.
After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/mistral_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF Mistral 7B checkpoint into FP8 and export a single-rank checkpoint
python quantize.py --model_dir mistralai/Mistral-7B-v0.1 \
                   --dtype float16 \
                   --qformat fp8 \
                   --export_path quantized_fp8 \
                   --calib_size 16

# Build Mistral 7B TP=8 using HF checkpoint + PTQ scaling factors from the single-rank checkpoint
python build.py --model_dir mistralai/Mistral-7B-v0.1 \
                --quantized_fp8_model_path ./quantized_fp8/mistral_tp1_rank0.npz \
                --dtype float16 \
                --enable_context_fmha \
                --use_gpt_attention_plugin float16 \
                --output_dir mistralai/Mistral-7B-v0.1/trt_engines/fp8/8-gpu/ \
                --remove_input_padding \
                --enable_fp8 \
                --fp8_kv_cache \
                --strongly_typed \
                --world_size 8 \
                --tp_size 8 \
                --load_by_shard \
                --parallel_build
```

### 4. Run

```bash
pip install -r requirements.txt
```

```bash
python summarize.py --test_trt_llm \
                    --hf_model_location mistral/rw-1b \
                    --data_type float16 \
                    --engine_dir mistral/rw-1b/trt_engines/fp16/1-gpu/

python summarize.py --test_trt_llm \
                      --hf_model_location mistral/7b-instruct \
                      --data_type bfloat16 \
                      --engine_dir mistral/7b-instruct/trt_engines/bf16/1-gpu

mpirun -n 2 --allow-run-as-root --oversubscribe \
    python summarize.py --test_trt_llm \
                      --hf_model_location mistral/40b-instruct \
                      --data_type bfloat16 \
                      --engine_dir mistral/40b-instruct/trt_engines/bf16/2-gpu
mpirun -n 8 --allow-run-as-root --oversubscribe \
    python summarize.py --test_trt_llm \
                      --hf_model_location mistral/180b \
                      --data_type bfloat16 \
                      --engine_dir mistral/180b/trt_engines/bf16/8-gpu
```

## Troubleshooting

### 1. The HuggingFace Mistral may raise an error when using  the `accelerate` package.

One may find the following message.
```
Traceback (most recent call last):
  File "build.py", line 10, in <module>
    from transformers import MistralConfig, MistralForCausalLM
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1090, in __getattr__
    value = getattr(module, name)
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1089, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1101, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.mistral.modeling_mistral because of the following error (look up to see its traceback):
```
It may be resolved by pinning the version of `typing-extensions` package by `4.5.0`.
```bash
pip install typing-extensions==4.5.0
```
