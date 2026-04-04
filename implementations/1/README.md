# Implementation #1: SDD + Antigravity

- [Implementation #1: SDD + Antigravity](#implementation-1-sdd--antigravity)
  - [Introduction](#introduction)
  - [Agents at work](#agents-at-work)
  - [Testing and debugging](#testing-and-debugging)
    - [Testbed](#testbed)
      - [Starting vLLM locally](#starting-vllm-locally)
      - [Starting Qdrant locally](#starting-qdrant-locally)
    - [Debugging](#debugging)

## Introduction

In order to push the project's AI-assisted philosophy as much as possible, the specifics were not written by hand. Rather, I sent the [design document](./Retriva-SDD_implementation_1.pdf) to MS Copilot so that it could write the spec files in accordance with Antigravity's conventions (interestingly, I had created the design document by brainstorming with Copilot itself!).

For security reasons, I installed a VirtualBox virtual machine running Ubuntu 24.04 on my development host — which runs Ubuntu 24.04 as well — and used Antigravity inside it. In this way, I sandboxed the agentic tool to prevent it from unintentionally compromising the host operating system. I used a condivisa directory to exchange files between the host and the guest:

```
# Shared folder seen from the guest machine:
llandre@vm-ubnt-24-04-4:~$ mount | grep vboxsf
implementation on /mnt/shared/implementation type vboxsf (rw,nodev,relatime,iocharset=utf8,uid=0,gid=984,dmode=0770,fmode=0770,tag=VBoxAutomounter)
```

## Agents at work

As usual, Copilot was zelous, therefore it has also produced instructions on how to use the ready-to-drop SDD pack it generated:

```
How to use it in Antigravity

Unzip the package into your project root so the .agent/ folder sits at workspace level. Community Antigravity workflow packs rely on .agent/workflows, and the slash commands are detected from those Markdown workflow files.

Open the workspace in Antigravity and start a Planning conversation; Antigravity recommends Planning mode for deeper, more complex work that should produce reviewable artifacts.
Run the workflows in this order:
* /define Finalize specs/001-qa-only-poc using the existing design package.
* /architect Prepare the approved implementation plan for specs/001-qa-only-poc.
* /execute Implement only the approved tasks in specs/001-qa-only-poc/tasks.md.
* /verify Validate specs/001-qa-only-poc against its acceptance criteria.
```

I followed the suggested flow without any particular issues.

This is how I configured the agent:

![](assets/screenshot_01.png)

Occasionally, I experienced this error, but I always solved it by triggering the operation again:

![](assets/screenshot_02.png)

Sometimes, I interacted with the agents during the execution of the workflow:

![](assets/screenshot_03.png)

This is the last step of the workflow:

![](assets/screenshot_04.png)

At some point in the workflow, I also executed some tests prepared by Antigravity to verify that what was coded until then was OK. I had to fix a couple of things to make test work, of course with the help of the conversational agent:

![](assets/screenshot_05.png)

## Testing and debugging

Even in the age of artificial intelligence, it seems that the testing and debugging process in the fields of computer and electronic engineering continues to play a crucial role in product development. In this section, I'll go over how I addressed this phase.

As planned, in this very early phase, I used local models for the sake of simplicity and cost-effectiveness. Performance was not a priority at this stage. Instead, the goal was to verify the system’s core functionalities. Therefore, I did not pay too much attention to the LLMs used.

### Testbed

The testbed is shown in the following image.

![](assets/20260404_140758_fully_local_testbed.drawio.png)

Note that I have set VM’s networking to bridged mode so that both machines (host and guest) have their own IP addresses and are connected to the same network. I disabled promiscuous mode on the virtual machine, however, for security reasons.

#### Starting vLLM locally

I struggled a little bit to start vLLM locally because I started using Qwen3-0.6B for testing. Although the model is relatively small, other default parameters made it unfeasible to run this LLM on my host, which is equipped with an NVIDIA GeForce RTX 3060 12GB. With the help of — guest what? — Copilot, I figured it out, however:

```
(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLM$ docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HF_TOKEN=$HF_TOKEN"     -p 8000:8000     --ipc=host     vllm/vllm-openai:latest     --model Qwen/Qwen3-0.6B
Unable to find image 'vllm/vllm-openai:latest' locally
latest: Pulling from vllm/vllm-openai
66587c81b81a: Pull complete
...
d3f429fd0a6f: Pull complete 
Digest: sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
Status: Downloaded newer image for vllm/vllm-openai:latest
WARNING 04-03 15:05:37 [argparse_utils.py:191] With `vllm serve`, you should provide the model as a positional argument or in a config file instead of via the `--model` option. The `--model` option will be removed in v0.13.
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299] 
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.19.0
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299]   █▄█▀ █     █     █     █  model   Qwen/Qwen3-0.6B
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:299] 
(APIServer pid=1) INFO 04-03 15:05:37 [utils.py:233] non-default args: {'model_tag': 'Qwen/Qwen3-0.6B'}
(APIServer pid=1) INFO 04-03 15:05:44 [model.py:549] Resolved architecture: Qwen3ForCausalLM
(APIServer pid=1) INFO 04-03 15:05:44 [model.py:1678] Using max model len 40960
(APIServer pid=1) INFO 04-03 15:05:44 [vllm.py:790] Asynchronous scheduling is enabled.
(EngineCore pid=229) INFO 04-03 15:05:56 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=229) INFO 04-03 15:05:56 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.3:48345 backend=nccl
(EngineCore pid=229) INFO 04-03 15:05:56 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=229) Process EngineCore:
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108] EngineCore failed to start.
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108] Traceback (most recent call last):
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1082, in run_engine_core
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     return func(*args, **kwargs)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 848, in __init__
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     super().__init__(
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 114, in __init__
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     self.model_executor = executor_class(vllm_config)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     return func(*args, **kwargs)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 103, in __init__
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     self._init_executor()
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/uniproc_executor.py", line 47, in _init_executor
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     self.driver_worker.init_device()
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py", line 312, in init_device
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     self.worker.init_device()  # type: ignore
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     return func(*args, **kwargs)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 283, in init_device
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     self.requested_memory = request_memory(init_snapshot, self.cache_config)
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py", line 413, in request_memory
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108]     raise ValueError(
(EngineCore pid=229) ERROR 04-03 15:05:56 [core.py:1108] ValueError: Free memory on device cuda:0 (9.02/11.63 GiB) on startup is less than desired GPU memory utilization (0.9, 10.47 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=229) Traceback (most recent call last):
(EngineCore pid=229)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore pid=229)     self.run()
(EngineCore pid=229)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore pid=229)     self._target(*self._args, **self._kwargs)
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1112, in run_engine_core
(EngineCore pid=229)     raise e
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1082, in run_engine_core
(EngineCore pid=229)     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=229)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229)     return func(*args, **kwargs)
(EngineCore pid=229)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 848, in __init__
(EngineCore pid=229)     super().__init__(
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 114, in __init__
(EngineCore pid=229)     self.model_executor = executor_class(vllm_config)
(EngineCore pid=229)                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229)     return func(*args, **kwargs)
(EngineCore pid=229)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 103, in __init__
(EngineCore pid=229)     self._init_executor()
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/uniproc_executor.py", line 47, in _init_executor
(EngineCore pid=229)     self.driver_worker.init_device()
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py", line 312, in init_device
(EngineCore pid=229)     self.worker.init_device()  # type: ignore
(EngineCore pid=229)     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=229)     return func(*args, **kwargs)
(EngineCore pid=229)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 283, in init_device
(EngineCore pid=229)     self.requested_memory = request_memory(init_snapshot, self.cache_config)
(EngineCore pid=229)                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=229)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py", line 413, in request_memory
(EngineCore pid=229)     raise ValueError(
(EngineCore pid=229) ValueError: Free memory on device cuda:0 (9.02/11.63 GiB) on startup is less than desired GPU memory utilization (0.9, 10.47 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
[rank0]:[W403 15:05:57.744526898 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(APIServer pid=1) Traceback (most recent call last):
(APIServer pid=1)   File "/usr/local/bin/vllm", line 10, in <module>
(APIServer pid=1)     sys.exit(main())
(APIServer pid=1)              ^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/main.py", line 75, in main
(APIServer pid=1)     args.dispatch_function(args)
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/serve.py", line 122, in cmd
(APIServer pid=1)     uvloop.run(run_server(args))
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=1)     return __asyncio.run(
(APIServer pid=1)            ^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/lib/python3.12/asyncio/runners.py", line 195, in run
(APIServer pid=1)     return runner.run(main)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=1)     return self._loop.run_until_complete(task)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=1)     return await main
(APIServer pid=1)            ^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 670, in run_server
(APIServer pid=1)     await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 684, in run_server_worker
(APIServer pid=1)     async with build_async_engine_client(
(APIServer pid=1)                ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=1)     return await anext(self.gen)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 100, in build_async_engine_client
(APIServer pid=1)     async with build_async_engine_client_from_engine_args(
(APIServer pid=1)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=1)     return await anext(self.gen)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 136, in build_async_engine_client_from_engine_args
(APIServer pid=1)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=1)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 225, in from_vllm_config
(APIServer pid=1)     return cls(
(APIServer pid=1)            ^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=1)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=1)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=1)     return func(*args, **kwargs)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 130, in make_async_mp_client
(APIServer pid=1)     return AsyncMPClient(*client_args)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=1)     return func(*args, **kwargs)
(APIServer pid=1)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 887, in __init__
(APIServer pid=1)     super().__init__(
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 535, in __init__
(APIServer pid=1)     with launch_core_engines(
(APIServer pid=1)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=1)   File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=1)     next(self.gen)
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 998, in launch_core_engines
(APIServer pid=1)     wait_for_engine_startup(
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1057, in wait_for_engine_startup
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

```
(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLM$ docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --enforce-eager

(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299] 
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.19.0
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299]   █▄█▀ █     █     █     █  model   Qwen/Qwen3-0.6B
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:299] 
(APIServer pid=1) INFO 04-03 17:24:25 [utils.py:233] non-default args: {'model_tag': 'Qwen/Qwen3-0.6B', 'max_model_len': 8192, 'enforce_eager': True, 'gpu_memory_utilization': 0.75}
(APIServer pid=1) INFO 04-03 17:24:33 [model.py:549] Resolved architecture: Qwen3ForCausalLM
(APIServer pid=1) INFO 04-03 17:24:33 [model.py:1678] Using max model len 8192
(APIServer pid=1) INFO 04-03 17:24:33 [vllm.py:790] Asynchronous scheduling is enabled.
(APIServer pid=1) WARNING 04-03 17:24:33 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(APIServer pid=1) WARNING 04-03 17:24:33 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=1) INFO 04-03 17:24:33 [vllm.py:1025] Cudagraph is disabled under eager mode
(APIServer pid=1) INFO 04-03 17:24:33 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=188) INFO 04-03 17:24:39 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=188) INFO 04-03 17:24:40 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.3:44903 backend=nccl
(EngineCore pid=188) INFO 04-03 17:24:40 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=188) INFO 04-03 17:24:40 [gpu_model_runner.py:4735] Starting to load model Qwen/Qwen3-0.6B...
(EngineCore pid=188) INFO 04-03 17:24:41 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=188) INFO 04-03 17:24:41 [flash_attn.py:596] Using FlashAttention version 2
(EngineCore pid=188) INFO 04-03 17:36:30 [weight_utils.py:581] Time spent downloading weights for Qwen/Qwen3-0.6B: 708.504603 seconds
(EngineCore pid=188) INFO 04-03 17:36:30 [weight_utils.py:625] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.30s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:02<00:00,  2.30s/it]
(EngineCore pid=188) 
(EngineCore pid=188) INFO 04-03 17:36:33 [default_loader.py:384] Loading weights took 2.31 seconds
(EngineCore pid=188) INFO 04-03 17:36:33 [gpu_model_runner.py:4820] Model loading took 1.12 GiB memory and 712.474857 seconds
(EngineCore pid=188) INFO 04-03 17:36:41 [gpu_worker.py:436] Available KV cache memory: 8.03 GiB
(EngineCore pid=188) INFO 04-03 17:36:41 [kv_cache_utils.py:1319] GPU KV cache size: 75,152 tokens
(EngineCore pid=188) INFO 04-03 17:36:41 [kv_cache_utils.py:1324] Maximum concurrency for 8,192 tokens per request: 9.17x
(EngineCore pid=188) ERROR 04-03 17:36:41 [core.py:1108] EngineCore failed to start.
(EngineCore pid=188) ERROR 04-03 17:36:41 [core.py:1108] Traceback (most recent call last):
(EngineCore pid=188) ERROR 04-03 17:36:41 [core.py:1108]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5596, in _dummy_sampler_run
...
(APIServer pid=1)   File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=1)     next(self.gen)
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 998, in launch_core_engines
(APIServer pid=1)     wait_for_engine_startup(
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1057, in wait_for_engine_startup
(APIServer pid=1)     raise RuntimeError(
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLM$ docker run -d --name vllm-qwen --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.60 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --enforce-eager
4fc9bb09d8a11bd1f05ef41f674f65c8e4230ba4c41cd166face710baa212858
(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLM$ docker logs -f vllm-qwen
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299] 
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.19.0
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299]   █▄█▀ █     █     █     █  model   Qwen/Qwen3-0.6B
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:299] 
(APIServer pid=1) INFO 04-03 17:49:21 [utils.py:233] non-default args: {'model_tag': 'Qwen/Qwen3-0.6B', 'max_model_len': 4096, 'enforce_eager': True, 'gpu_memory_utilization': 0.6, 'max_num_batched_tokens': 1024, 'max_num_seqs': 8}
(APIServer pid=1) INFO 04-03 17:49:27 [model.py:549] Resolved architecture: Qwen3ForCausalLM
(APIServer pid=1) INFO 04-03 17:49:27 [model.py:1678] Using max model len 4096
(APIServer pid=1) INFO 04-03 17:49:27 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=1024.
(APIServer pid=1) INFO 04-03 17:49:27 [vllm.py:790] Asynchronous scheduling is enabled.
(APIServer pid=1) WARNING 04-03 17:49:27 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(APIServer pid=1) WARNING 04-03 17:49:27 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=1) INFO 04-03 17:49:27 [vllm.py:1025] Cudagraph is disabled under eager mode
(APIServer pid=1) INFO 04-03 17:49:27 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=188) INFO 04-03 17:49:33 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='Qwen/Qwen3-0.6B', speculative_config=None, tokenizer='Qwen/Qwen3-0.6B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3-0.6B, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [1024], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=188) INFO 04-03 17:49:34 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.3:55541 backend=nccl
(EngineCore pid=188) INFO 04-03 17:49:34 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=188) INFO 04-03 17:49:34 [gpu_model_runner.py:4735] Starting to load model Qwen/Qwen3-0.6B...
(EngineCore pid=188) INFO 04-03 17:49:35 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=188) INFO 04-03 17:49:35 [flash_attn.py:596] Using FlashAttention version 2
(EngineCore pid=188) INFO 04-03 17:49:36 [weight_utils.py:625] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.46s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.46s/it]
(EngineCore pid=188) 
(EngineCore pid=188) INFO 04-03 17:49:38 [default_loader.py:384] Loading weights took 1.49 seconds
(EngineCore pid=188) INFO 04-03 17:49:38 [gpu_model_runner.py:4820] Model loading took 1.12 GiB memory and 3.162604 seconds
(EngineCore pid=188) INFO 04-03 17:49:45 [gpu_worker.py:436] Available KV cache memory: 5.68 GiB
(EngineCore pid=188) INFO 04-03 17:49:45 [kv_cache_utils.py:1319] GPU KV cache size: 53,184 tokens
(EngineCore pid=188) INFO 04-03 17:49:45 [kv_cache_utils.py:1324] Maximum concurrency for 4,096 tokens per request: 12.98x
(EngineCore pid=188) INFO 04-03 17:49:46 [core.py:283] init engine (profile, create kv cache, warmup model) took 7.52 seconds
(EngineCore pid=188) INFO 04-03 17:49:47 [vllm.py:790] Asynchronous scheduling is enabled.
(EngineCore pid=188) WARNING 04-03 17:49:47 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=188) WARNING 04-03 17:49:47 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=188) INFO 04-03 17:49:47 [vllm.py:1025] Cudagraph is disabled under eager mode
(EngineCore pid=188) INFO 04-03 17:49:47 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=1) INFO 04-03 17:49:47 [api_server.py:590] Supported tasks: ['generate']
(APIServer pid=1) WARNING 04-03 17:49:47 [model.py:1435] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1) INFO 04-03 17:49:50 [hf.py:314] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1) INFO 04-03 17:49:50 [api_server.py:594] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:37] Available routes are:
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=1) INFO 04-03 17:49:50 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

Some health checks:

```
llandre@llandre0:~$ curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"Qwen/Qwen3-0.6B","object":"model","created":1775238711,"owned_by":"vllm","root":"Qwen/Qwen3-0.6B","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-912d7b6ba7fbecfb","object":"model_permission","created":1775238711,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}


llandre@llandre0:~$ curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
{"id":"chatcmpl-a6769d62d79b8cf1","object":"chat.completion","created":1775238891,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, the user wants a one-sentence hello. Let me think about the simplest way to say \"hello\" in one sentence. Maybe start with \"Hello, \" and add a greeting. Let me check the grammar. \"Hello, how are you?\" sounds good. It's friendly and includes a question. That should work. Let me make sure there's no extra words. Yep, that's concise. Alright, that's the answer.\n</think>\n\nHello, how are you?","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":14,"total_tokens":115,"completion_tokens":101,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}

llandre@llandre0:~$ curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0
  }'
{"id":"chatcmpl-a6bdc32c8c0b5d4e","object":"chat.completion","created":1775239134,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, the user wants a one-sentence greeting. Let me think. They probably need something friendly and simple. Maybe start with \"Hello\" and add a few words. Let me check if it's concise. \"Hello, how are you today?\" That works. It's direct and includes a question. I should make sure there's no extra words. Yep, that's one sentence. Let me double-check for any grammatical errors. Looks good. Alright, that's the response.\n</think>\n\nHello, how are you today?","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":14,"total_tokens":125,"completion_tokens":111,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

When I switched to a smaller, embedding-optimized model, it worked without tweaking any parameters:

```
llandre@llandre0:~/devel/ai/retriva/implementation/1/retriva-v0.1$ docker run --name vllm-ibm-granite-embedding-english-r2 --gpus all   -v ~/.cache/huggingface:/root/.cache/huggingface   -e HF_TOKEN="$HF_TOKEN"   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   -p 8000:8000   --ipc=host   vllm/vllm-openai:latest ibm-granite/granite-embedding-english-r2
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299] 
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.19.0
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299]   █▄█▀ █     █     █     █  model   ibm-granite/granite-embedding-english-r2
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:299] 
(APIServer pid=1) INFO 04-04 09:14:16 [utils.py:233] non-default args: {'model_tag': 'ibm-granite/granite-embedding-english-r2', 'model': 'ibm-granite/granite-embedding-english-r2'}
(APIServer pid=1) INFO 04-04 09:14:18 [config.py:945] Found sentence-transformers tokenize configuration.
(APIServer pid=1) INFO 04-04 09:14:23 [config.py:833] Found sentence-transformers modules configuration.
(APIServer pid=1) INFO 04-04 09:14:23 [config.py:860] Found pooling configuration.
(APIServer pid=1) INFO 04-04 09:14:23 [model.py:864] Resolved `--runner auto` to `--runner pooling`. Pass the value explicitly to silence this message.
(APIServer pid=1) INFO 04-04 09:14:23 [model.py:916] Resolved `--convert auto` to `--convert embed`. Pass the value explicitly to silence this message.
(APIServer pid=1) INFO 04-04 09:14:23 [model.py:549] Resolved architecture: ModernBertModel
(APIServer pid=1) INFO 04-04 09:14:24 [model.py:1678] Using max model len 8192
(APIServer pid=1) INFO 04-04 09:14:24 [vllm.py:790] Asynchronous scheduling is enabled.
(APIServer pid=1) WARNING 04-04 09:14:24 [vllm.py:977] Pooling models do not support full cudagraphs. Overriding cudagraph_mode to PIECEWISE.
(EngineCore pid=191) INFO 04-04 09:14:31 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='ibm-granite/granite-embedding-english-r2', speculative_config=None, tokenizer='ibm-granite/granite-embedding-english-r2', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=ibm-granite/granite-embedding-english-r2, enable_prefix_caching=False, enable_chunked_prefill=False, pooler_config=PoolerConfig(task=None, pooling_type=None, seq_pooling_type='CLS', tok_pooling_type='ALL', use_activation=False, dimensions=None, enable_chunked_processing=False, max_embed_len=None, logit_bias=None, step_tag_id=None, returned_token_ids=None), compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=191) INFO 04-04 09:14:31 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.2:53623 backend=nccl
(EngineCore pid=191) INFO 04-04 09:14:31 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=191) INFO 04-04 09:14:32 [gpu_model_runner.py:4735] Starting to load model ibm-granite/granite-embedding-english-r2...
(EngineCore pid=191) INFO 04-04 09:14:32 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=191) INFO 04-04 09:14:32 [flash_attn.py:596] Using FlashAttention version 2
(EngineCore pid=191) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
(EngineCore pid=191) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
(EngineCore pid=191) INFO 04-04 09:17:55 [weight_utils.py:581] Time spent downloading weights for ibm-granite/granite-embedding-english-r2: 201.441856 seconds
(EngineCore pid=191) INFO 04-04 09:17:55 [weight_utils.py:625] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 429.08it/s]
(EngineCore pid=191) 
(EngineCore pid=191) INFO 04-04 09:17:55 [default_loader.py:384] Loading weights took 0.05 seconds
(EngineCore pid=191) INFO 04-04 09:17:55 [gpu_model_runner.py:4820] Model loading took 0.28 GiB memory and 202.704861 seconds
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s](EngineCore pid=191) INFO 04-04 09:17:58 [backends.py:1051] Using cache directory: /root/.cache/vllm/torch_compile_cache/3f706156bc/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=191) INFO 04-04 09:17:58 [backends.py:1111] Dynamo bytecode transform time: 2.33 s
[rank0]:W0404 09:17:58.809000 191 torch/_inductor/utils.py:1679] Not enough SMs to use max_autotune_gemm mode
(EngineCore pid=191) INFO 04-04 09:18:00 [backends.py:372] Cache the graph of compile range (1, 8192) for later use
(EngineCore pid=191) INFO 04-04 09:18:02 [backends.py:390] Compiling a graph for compile range (1, 8192) takes 4.02 s
(EngineCore pid=191) INFO 04-04 09:18:02 [decorators.py:640] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/7b2d3fbedcf998eb00c504e4190139579ca7dceddd18bbc5fb59ad97cf017955/rank_0_0/model
(EngineCore pid=191) INFO 04-04 09:18:02 [monitor.py:48] torch.compile took 7.01 s in total
(EngineCore pid=191) INFO 04-04 09:18:03 [monitor.py:76] Initial profiling/warmup run took 0.28 s
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 51/51 [00:08<00:00,  6.33it/s]
(EngineCore pid=191) INFO 04-04 09:18:04 [gpu_model_runner.py:6046] Graph capturing finished in 8 secs, took 0.25 GiB
(EngineCore pid=191) INFO 04-04 09:18:04 [core.py:283] init engine (profile, create kv cache, warmup model) took 8.50 seconds
(EngineCore pid=191) INFO 04-04 09:18:04 [config.py:945] Found sentence-transformers tokenize configuration.
(EngineCore pid=191) INFO 04-04 09:18:04 [vllm.py:790] Asynchronous scheduling is enabled.
(APIServer pid=1) INFO 04-04 09:18:04 [api_server.py:590] Supported tasks: ['token_embed', 'embed']
(APIServer pid=1) WARNING 04-04 09:18:04 [utils.py:132] To make v1/embeddings API fast, please install orjson by `pip install orjson`
(APIServer pid=1) INFO 04-04 09:18:05 [api_server.py:594] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:37] Available routes are:
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /docs, Methods: GET, HEAD
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /pooling, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v1/embeddings, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v2/embed, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /score, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v1/score, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /rerank, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v1/rerank, Methods: POST
(APIServer pid=1) INFO 04-04 09:18:05 [launcher.py:46] Route: /v2/rerank, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.

llandre@llandre0:~/devel/ai/retriva/implementation/1/retriva-v0.1$ nvidia-smi
Sat Apr  4 11:28:30 2026     
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:05:00.0  On |                  N/A |
|  0%   47C    P8             18W /  170W |    1587MiB /  12288MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           10096      G   /usr/lib/xorg/Xorg                      504MiB |
|    0   N/A  N/A           10897      G   cinnamon                                 38MiB |
|    0   N/A  N/A           13324      G   .../7967/usr/lib/firefox/firefox        194MiB |
|    0   N/A  N/A           64683      G   /usr/share/code/code                     54MiB |
|    0   N/A  N/A           82022      G   ...6932/usr/bin/telegram-desktop          2MiB |
|    0   N/A  N/A          130565      C   VLLM::EngineCore                        720MiB |
+-----------------------------------------------------------------------------------------+
```

#### Starting Qdrant locally
Qdrant started without any problem at the first attempt:

```
(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLMdocker pull qdrant/qdrantnt
Using default tag: latest
latest: Pulling from qdrant/qdrant
ec781dee3f47: Pull complete 
4f4fb700ef54: Pull complete 
ecadb5c5abda: Pull complete 
46d6a2dfacae: Pull complete 
37d19898a8e4: Pull complete 
4b01f989bd9b: Pull complete 
be9ea85b506c: Pull complete 
e7b33762354c: Pull complete 
Digest: sha256:94728574965d17c6485dd361aa3c0818b325b9016dac5ea6afec7b4b2700865f
Status: Downloaded newer image for qdrant/qdrant:latest
docker.io/qdrant/qdrant:latest
(vLLM) llandre@llandre0:~/devel/ai/retriva/implementation/vLLM$ docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
           _                 _  
  __ _  __| |_ __ __ _ _ __ | |_  
 / _` |/ _` | '__/ _` | '_ \| __| 
| (_| | (_| | | | (_| | | | | |_  
 \__, |\__,_|_|  \__,_|_| |_|\__| 
    |_|                         

Version: 1.17.1, build: eabee371
Access web UI at http://localhost:6333/dashboard

2026-04-03T18:09:40.944167Z  INFO storage::content_manager::consensus::persistent: Initializing new raft state at ./storage/raft_state.json
2026-04-03T18:09:40.958291Z  INFO qdrant: Distributed mode disabled
2026-04-03T18:09:40.958311Z  INFO qdrant: Telemetry reporting enabled, id: ec305c19-99f4-4bb4-95a5-9bedf22e27f2
2026-04-03T18:09:40.970821Z  INFO qdrant::actix: REST transport settings: keep_alive=5s, client_request_timeout=5s, client_disconnect_timeout=5s
2026-04-03T18:09:40.970834Z  INFO qdrant::actix: TLS disabled for REST API
2026-04-03T18:09:40.970879Z  INFO qdrant::actix: Qdrant HTTP listening on 6333
2026-04-03T18:09:40.970884Z  INFO actix_server::builder: starting 31 workers
2026-04-03T18:09:40.970887Z  INFO actix_server::server: Actix runtime found; starting in Actix runtime
2026-04-03T18:09:40.970892Z  INFO actix_server::server: starting service: "actix-web-service-0.0.0.0:6333", workers: 31, listening on: 0.0.0.0:6333
2026-04-03T18:09:40.972049Z  INFO qdrant::tonic: Qdrant gRPC listening on 6334
2026-04-03T18:09:40.972060Z  INFO qdrant::tonic: TLS disabled for gRPC API
```
### Debugging
Since Antigravity is a customized version of VSCode and I'm already familiar with using this IDE, there's practically no learning curve when it comes to performing typical tasks like debugging code.

TBD
