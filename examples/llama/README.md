# LLaMA

This document shows how to build and run a LLaMA model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM LLaMA implementation can be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py). The TensorRT-LLM LLaMA example code is located in [`examples/llama`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the LLaMA model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

The TensorRT-LLM LLaMA example code locates at [examples/llama](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF LLaMA checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/llama.

TensorRT-LLM LLaMA builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in LLaMA.
# Try use_gemm_plugin to prevent accuracy issue.

# Build the LLaMA 7B model using a single GPU and FP16.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Build the LLaMA 7B model using a single GPU and BF16.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --use_gemm_plugin bfloat16 \
                --output_dir ./tmp/llama/7B/trt_engines/bf16/1-gpu/

# Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --output_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/

# Build LLaMA 7B using 2-way tensor parallelism.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ \
                --world_size 2

# Build LLaMA 30B using 2-way tensor parallelism.
python build.py --model_dir ./tmp/llama/30B/hf/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/ \
                --world_size 2
```

#### LLaMA v2 Updates
The LLaMA v2 models with 7B and 13B are compatible with the LLaMA v1 implementation. The above
commands still work.

~~For LLaMA v2 70B, the current implementation in TensorRT-LLM requires the number of KV heads to
match the number of GPUs. It means that LLaMA v2 70B requires 8 GPUs (8-way Tensor Parallelism)
to work. That limitation will be removed in a future version of TensorRT-LLM.~~

UPDATE: LLaMA v2 70B now works with less than 8 GPUs. Current restriction is that the number of KV heads
must be **divisible by the number of GPUs**. For example, since the 70B model has 8 KV heads, you can run it with
2, 4 or 8 GPUs (even 1 GPU once FP8 is supported).


```bash
# Build LLaMA 70B using 8-way tensor parallelism.
python build.py --model_dir ./tmp/llama/70B/hf/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8


# Build LLaMA 70B TP=8 using Meta checkpoints directly.
python build.py --meta_ckpt_dir ./tmp/llama/70B \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8
```

Same instructions can be applied to fine-tuned versions of the LLaMA v2 models (e.g. 7Bf or llama-2-7b-chat).

#### SmoothQuant

The smoothquant supports both LLaMA v1 and LLaMA v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_llama_convert.py -i /llama-models/llama-7b-hf -o /llama/smooth_llama_7B/sq0.8/ -sq 0.8 --tensor-parallelism 1 --storage-type fp16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --ft_model_dir=/llama/smooth_llama_7B/sq0.8/1-gpu/ \
                 --use_smooth_quant

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --ft_model_dir=/llama/smooth_llama_7B/sq0.8/1-gpu/ \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel
```

Note we use `--ft_model_dir` instead of `--model_dir` and `--meta_ckpt_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

#### Groupwise quantization (AWQ/GPTQ)

To run the GPTQ LLaMa example, the following steps are required:

1. Generating quantized weights using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa.git):

```bash
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
cd GPTQ-for-LLaMa

# Quantize weights into INT4 and save as safetensors
# Quantized weight with parameter "--act-order" is not supported in TRT-LLM
python llama.py ./tmp/llama/7B/ c4 --wbits 4 --true-sequential --groupsize 128 --save_safetensors ./llama-7b-4bit-gs128.safetensors
```
2. Build TRT-LLM engine:

    [`build.py`](./build.py) add "--per_group" for the support of INT4 per-group quantization and "--quant_safetensors_path" for linking generated safetensors.


```python
# Build the LLaMA 7B model using a single GPU and apply INT4 GPTQ quantization.
# Compressed checkpoint safetensors are generated seperately from GPTQ.
python build.py --model_dir ./tmp/llama/7B/ \
                --quant_safetensors_path ./llama-7b-4bit-gs128.safetensors \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --per_group \
                --output_dir ./tmp/llama/7B/trt_engines/int4_GPTQ/1-gpu/

# Build the LLaMA 7B model using 2-way tensor parallelism and apply INT4 GPTQ quantization.
# Compressed checkpoint safetensors are generated seperately from GPTQ.
python build.py --model_dir ./tmp/llama/7B/ \
                --quant_safetensors_path ./llama-7b-4bit-gs128.safetensors \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --per_group \
                --world_size 2 \
                --output_dir ./tmp/llama/7B/trt_engines/int4_GPTQ/2-gpu/
```

### Run

To run a TensorRT-LLM LLaMA model using the engines generated by build.py

```bash
# With fp16 inference
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/bf16/1-gpu/
```

### Summarization using the LLaMA model

```bash
# Run summarization using the LLaMA 7B model in FP16.
python summarize.py --test_trt_llm \
                    --hf_model_location ./tmp/llama/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Run summarization using the LLaMA 7B model quantized to INT8.
python summarize.py --test_trt_llm \
                    --hf_model_location ./tmp/llama/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/

# Run summarization using the LLaMA 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./tmp/llama/7B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/

# Run summarization using the LLaMA 30B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./tmp/llama/30B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/
```