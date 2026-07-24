"""Real-world test log fixtures based on actual llama.cpp / LocalAI logs found online."""

from __future__ import annotations


def llama_cpp_cli_log() -> str:
    """Classic llama.cpp CLI output (llama_print_timings format).
    
    Source: Various GitHub issues and blog posts showing CLI output.
    """
    return """llama_print_timings:        load time =   3436.49 ms

llama_print_timings:      sample time =     30.06 ms /   12 runs   (    2.51 ms per token,   399.16 tokens per second)
llama_print_timings: prompt eval time =   3432.49 ms /   4472 tokens (    0.77 ms per token,   1302.84 tokens per second)
llama_print_timings:        eval time =  56821.30 ms /   149 runs   (  381.35 ms per token,     2.62 tokens per second)
llama_print_timings:       total time =  59050.97 ms /   149 tokens
"""


def llama_cpp_server_full_log() -> str:
    """Complete server log with multiple tasks, speculative decoding, and interleaved output.
    
    Based on real LocalAI / llama.cpp server logs with speculative decoding.
    """
    return """0.07.587.079 I srv  PredictStrea: [TOOLS DEBUG] PredictStream: has_grammar_from_go=0, data.contains("tools")=0, data.contains("grammar")=0
0.07.587.080 W srv  PredictStrea: No tools found in data - tool calls will not work without tools field
0.07.587.141 I srv  PredictStrea: [CONTENT DEBUG] PredictStream: Before oaicompat_chat_params_parse - checking 46 messages
0.07.608.225 I start_llama_server: model loaded
0.07.657.508 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.07.657.589 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.10.754.624 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   2048, progress = 0.60, t =   3.10 s / 661.28 tokens per second
0.11.571.293 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   2560, progress = 0.75, t =   3.91 s / 654.11 tokens per second
0.12.377.751 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   2900, progress = 0.85, t =   4.72 s / 614.39 tokens per second
0.13.103.163 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =   3408, progress = 1.00, t =   5.45 s / 625.83 tokens per second
0.14.412.071 I slot print_timing: id  0 | task 0 | prompt eval time =    6174.57 ms /  3412 tokens (    1.81 ms per token,   552.59 tokens per second)
0.14.412.080 I slot print_timing: id  0 | task 0 |        eval time =     579.82 ms /    15 tokens (   38.65 ms per token,    25.87 tokens per second)
0.14.412.082 I slot print_timing: id  0 | task 0 |       total time =    6754.39 ms /  3427 tokens
0.14.412.082 I slot print_timing: id  0 | task 0 |    graphs reused =         14
0.14.414.515 I slot      release: id  0 | task 0 | stop processing: n_tokens = 3426, truncated = 0
I slot update_slots: id 0 | task -1 | all slots are idle
0.15.000.100 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 14414
0.15.000.200 I slot launch_slot_: id  0 | task 1 | processing task, is_child = 0
0.15.500.300 I slot print_timing: id  0 | task 1 | prompt eval time =    2100.50 ms /  1050 tokens (    2.00 ms per token,   500.00 tokens per second)
0.15.800.400 I slot print_timing: id  0 | task 1 |        eval time =    4200.60 ms /   210 tokens (   20.00 ms per token,    50.00 tokens per second)
0.15.800.410 I slot print_timing: id  0 | task 1 |       total time =    6301.10 ms /  1260 tokens
0.15.800.420 I slot print_timing: id  0 | task 1 |    graphs reused =         50
0.15.800.430 I slot print_timing: id  0 | task 1 | draft acceptance = 0.45000 ( 90 accepted / 200 generated), mean len = 4.50
0.15.805.000 I slot      release: id  0 | task 1 | stop processing: n_tokens = 1260, truncated = 0
"""


def llama_cpp_server_speculative_decoding_log() -> str:
    """Server log with speculative decoding enabled.
    
    Based on real logs from llama.cpp issues with MTP and NGRAM speculative decoding.
    """
    return """0.01.000.000 I srv    load_model: loading model '/models/model.gguf'
0.05.000.000 I srv    load_model: model loaded successfully
0.05.100.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.05.100.100 I slot launch_slot_: id  0 | task 100 | processing task, is_child = 0
0.06.200.000 I slot print_timing: id  0 | task 100 | prompt eval time =    1100.00 ms /   550 tokens (    2.00 ms per token,   500.00 tokens per second)
0.06.500.000 I slot print_timing: id  0 | task 100 |        eval time =    5500.00 ms /   275 tokens (   20.00 ms per token,    50.00 tokens per second)
0.06.500.010 I slot print_timing: id  0 | task 100 |       total time =    6600.00 ms /   825 tokens
0.06.500.020 I slot print_timing: id  0 | task 100 |    graphs reused =         25
0.06.500.030 I slot print_timing: id  0 | task 100 | draft acceptance = 0.74177 ( 1126 accepted / 1518 generated), mean len = 5.85
0.06.510.000 I slot      release: id  0 | task 100 | stop processing: n_tokens = 825, truncated = 0
I slot update_slots: id 0 | task -1 | all slots are idle
0.07.000.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 6510
0.07.000.100 I slot launch_slot_: id  0 | task 200 | processing task, is_child = 0
0.08.100.000 I slot print_timing: id  0 | task 200 | prompt eval time =    1200.00 ms /   600 tokens (    2.00 ms per token,   500.00 tokens per second)
0.08.600.000 I slot print_timing: id  0 | task 200 |        eval time =    6000.00 ms /   300 tokens (   20.00 ms per token,    50.00 tokens per second)
0.08.600.010 I slot print_timing: id  0 | task 200 |       total time =    7200.00 ms /   900 tokens
0.08.600.020 I slot print_timing: id  0 | task 200 |    graphs reused =         30
0.08.600.030 I slot print_timing: id  0 | task 200 | draft acceptance = 0.65000 ( 195 accepted / 300 generated), mean len = 3.90
0.08.610.000 I slot      release: id  0 | task 200 | stop processing: n_tokens = 900, truncated = 0
"""


def llama_cpp_prefix_match_log() -> str:
    """Server log with KV-cache reuse (prefix-match hit).
    
    Based on real logs showing prompt cache reuse when similar prompts arrive.
    """
    return """0.01.000.000 I start_llama_server: model loaded
0.01.100.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.01.100.100 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =    2000.00 ms /  1000 tokens (    2.00 ms per token,   500.00 tokens per second)
0.02.300.000 I slot print_timing: id  0 | task 0 |        eval time =    1500.00 ms /    75 tokens (   20.00 ms per token,    50.00 tokens per second)
0.02.300.010 I slot print_timing: id  0 | task 0 |       total time =    3500.00 ms /  1075 tokens
0.02.300.020 I slot print_timing: id  0 | task 0 |    graphs reused =         10
0.02.310.000 I slot      release: id  0 | task 0 | stop processing: n_tokens = 1075, truncated = 0
Llama.generate: prefix-match hit
0.03.000.000 I slot print_timing: id  0 | task 1 | prompt eval time =       0.00 ms /     0 tokens (    -nan ms per token,     -nan tokens per second)
0.03.300.000 I slot print_timing: id  0 | task 1 |        eval time =    1550.00 ms /    77 tokens (   20.13 ms per token,    49.68 tokens per second)
0.03.300.010 I slot print_timing: id  0 | task 1 |       total time =    1550.00 ms /    77 tokens
0.03.310.000 I slot      release: id  0 | task 1 | stop processing: n_tokens = 77, truncated = 0
Llama.generate: prefix-match hit
0.04.000.000 I slot print_timing: id  0 | task 2 | prompt eval time =       0.00 ms /     0 tokens (    -nan ms per token,     -nan tokens per second)
0.04.250.000 I slot print_timing: id  0 | task 2 |        eval time =    1400.00 ms /    70 tokens (   20.00 ms per token,    50.00 tokens per second)
0.04.250.010 I slot print_timing: id  0 | task 2 |       total time =    1400.00 ms /    70 tokens
0.04.260.000 I slot      release: id  0 | task 2 | stop processing: n_tokens = 70, truncated = 0
"""


def llama_cpp_parallel_requests_log() -> str:
    """Server log simulating parallel requests from multiple slots.
    
    Based on benchmark logs showing concurrent request handling.
    """
    return """0.01.000.000 I start_llama_server: model loaded
0.01.100.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.01.100.001 I slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = -1
0.01.100.100 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.01.100.200 I slot launch_slot_: id  1 | task 1 | processing task, is_child = 0
0.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =    1500.00 ms /   750 tokens (    2.00 ms per token,   500.00 tokens per second)
0.02.100.000 I slot print_timing: id  1 | task 1 | prompt eval time =    1600.00 ms /   800 tokens (    2.00 ms per token,   500.00 tokens per second)
0.02.300.000 I slot print_timing: id  0 | task 0 |        eval time =    3000.00 ms /   150 tokens (   20.00 ms per token,    50.00 tokens per second)
0.02.400.000 I slot print_timing: id  1 | task 1 |        eval time =    3200.00 ms /   160 tokens (   20.00 ms per token,    50.00 tokens per second)
0.02.300.010 I slot print_timing: id  0 | task 0 |       total time =    4500.00 ms /   900 tokens
0.02.400.010 I slot print_timing: id  1 | task 1 |       total time =    4800.00 ms /   960 tokens
0.02.300.020 I slot print_timing: id  0 | task 0 |    graphs reused =          5
0.02.400.020 I slot print_timing: id  1 | task 1 |    graphs reused =          5
0.02.310.000 I slot      release: id  0 | task 0 | stop processing: n_tokens = 900, truncated = 0
0.02.410.000 I slot      release: id  1 | task 1 | stop processing: n_tokens = 960, truncated = 1
0.03.000.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 2310
0.03.000.100 I slot launch_slot_: id  0 | task 2 | processing task, is_child = 0
0.03.800.000 I slot print_timing: id  0 | task 2 | prompt eval time =    1400.00 ms /   700 tokens (    2.00 ms per token,   500.00 tokens per second)
0.04.100.000 I slot print_timing: id  0 | task 2 |        eval time =    2500.00 ms /   125 tokens (   20.00 ms per token,    50.00 tokens per second)
0.04.100.010 I slot print_timing: id  0 | task 2 |       total time =    3900.00 ms /   825 tokens
0.04.100.020 I slot print_timing: id  0 | task 2 |    graphs reused =         10
0.04.110.000 I slot      release: id  0 | task 2 | stop processing: n_tokens = 825, truncated = 0
"""


def llama_cpp_gpu_offload_log() -> str:
    """Server log with GPU offload and speculative decoding.
    
    Based on real logs from RTX 6000 servers with Qwen 27B model.
    """
    return """0.01.000.000 I srv    load_model: loading model '/models/Qwen2.5-27B.gguf'
0.01.500.000 I ggml_backend_cuda: GPU offload successful: 47/49 layers
0.02.000.000 I slot get_availabl: id  2 | task -1 | selected slot by LRU, t_last = -1
0.02.000.100 I slot launch_slot_: id  2 | task 1768 | processing task, is_child = 0
0.03.100.000 I slot print_timing: id  2 | task 1768 | prompt eval time =    2075.52 ms /    18 tokens (  115.31 ms per token,     8.67 tokens per second)
0.04.500.000 I slot print_timing: id  2 | task 1768 |        eval time =   18013.80 ms /   128 tokens (  140.73 ms per token,     7.11 tokens per second)
0.04.500.010 I slot print_timing: id  2 | task 1768 |       total time =   20089.32 ms /   146 tokens
0.04.500.020 I slot print_timing: id  2 | task 1768 |    graphs reused =          3
0.04.500.030 I slot print_timing: id  2 | task 1768 | draft acceptance = 0.94156 ( 7556 accepted / 8026 generated), mean len = 9.42
0.04.520.000 I slot      release: id  2 | task 1768 | stop processing: n_tokens = 146, truncated = 0
"""


def llama_cpp_dflash_log() -> str:
    """Server log with DFlash speculative decoding.
    
    Based on real logs from ROCm + DFlash setup.
    """
    return """0.01.000.000 I srv    load_model: loading model '/models/Qwen3.6-27B-DFlash.gguf'
0.01.500.000 I srv    load_model: model loaded
0.02.000.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.02.000.100 I slot launch_slot_: id  0 | task 56 | processing task, is_child = 0
0.02.200.000 I slot print_timing: id  0 | task 56 | prompt eval time =     213.87 ms /    17 tokens (   12.58 ms per token,   79.49 tokens per second)
0.04.000.000 I slot print_timing: id  0 | task 56 |        eval time =  152296.66 ms /   886 tokens (  171.89 ms per token,     5.82 tokens per second)
0.04.000.010 I slot print_timing: id  0 | task 56 |       total time =  152510.54 ms /   903 tokens
0.04.000.020 I slot print_timing: id  0 | task 56 |    graphs reused =          1
0.04.000.030 I slot print_timing: id  0 | task 56 | draft acceptance rate = 0.09722 ( 86 accepted / 885 generated), mean len = 1.05
0.04.020.000 I slot      release: id  0 | task 56 | stop processing: n_tokens = 903, truncated = 0
"""


def llama_cpp_system_info_log() -> str:
    """Server startup log with system info and model loading.
    
    Contains system information lines that should be ignored by the parser.
    """
    return """0.01.000.000 I system_info: n_threads = 12, n_threads_batch = 9, total_threads = 12
0.01.000.001 I 
0.01.000.010 I system_info: n_threads = 12 (n_threads_batch = 9) / 12 | ROCm : NO_VMM = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
0.01.000.011 I 
0.01.000.100 I srv    load_model: loading model '/models/model.gguf'
0.02.000.000 W srv    load_model: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.03.000.000 I srv    load_model: cache_reuse is not supported by this context, it will be disabled
0.03.500.000 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 105216, kv_unified = 'true'
0.04.000.000 I srv          init: chat template supports preserving reasoning, consider enabling it via --reasoning-preserve
0.04.500.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.04.500.100 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.05.500.000 I slot print_timing: id  0 | task 0 | prompt eval time =    1000.00 ms /   500 tokens (    2.00 ms per token,   500.00 tokens per second)
0.06.000.000 I slot print_timing: id  0 | task 0 |        eval time =    2000.00 ms /   100 tokens (   20.00 ms per token,    50.00 tokens per second)
0.06.000.010 I slot print_timing: id  0 | task 0 |       total time =    3000.00 ms /   600 tokens
0.06.000.020 I slot print_timing: id  0 | task 0 |    graphs reused =         10
0.06.010.000 I slot      release: id  0 | task 0 | stop processing: n_tokens = 600, truncated = 0
"""


def llama_cpp_truncated_log() -> str:
    """Log with truncated output (max tokens reached)."""
    return """0.01.000.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.01.000.100 I slot launch_slot_: id  0 | task 999 | processing task, is_child = 0
0.02.000.000 I slot print_timing: id  0 | task 999 | prompt eval time =    1000.00 ms /   500 tokens (    2.00 ms per token,   500.00 tokens per second)
0.05.000.000 I slot print_timing: id  0 | task 999 |        eval time =   3000.00 ms /  1000 tokens (    3.00 ms per token,  333.33 tokens per second)
0.05.000.010 I slot print_timing: id  0 | task 999 |       total time =    4000.00 ms /  1500 tokens
0.05.000.020 I slot print_timing: id  0 | task 999 |    graphs reused =         10
0.05.010.000 I slot      release: id  0 | task 999 | stop processing: n_tokens = 1500, truncated = 1
"""


def llama_cpp_zero_prompt_cache_hit_log() -> str:
    """Log with zero-prompt cache hit (prefix match with 0 tokens)."""
    return """0.01.000.000 I start_llama_server: model loaded
0.01.100.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.01.100.100 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =    2000.00 ms /  1000 tokens (    2.00 ms per token,   500.00 tokens per second)
0.02.300.000 I slot print_timing: id  0 | task 0 |        eval time =    1500.00 ms /    75 tokens (   20.00 ms per token,    50.00 tokens per second)
0.02.300.010 I slot print_timing: id  0 | task 0 |       total time =    3500.00 ms /  1075 tokens
0.02.310.000 I slot      release: id  0 | task 0 | stop processing: n_tokens = 1075, truncated = 0
I slot update_slots: id 0 | task -1 | all slots are idle
Llama.generate: prefix-match hit
0.03.000.000 I slot print_timing: id  0 | task 1 | prompt eval time =       0.00 ms /     0 tokens (    -nan ms per token,     -nan tokens per second)
0.03.300.000 I slot print_timing: id  0 | task 1 |        eval time =    1550.00 ms /    77 tokens (   20.13 ms per token,    49.68 tokens per second)
0.03.300.010 I slot print_timing: id  0 | task 1 |       total time =    1550.00 ms /    77 tokens
0.03.310.000 I slot      release: id  0 | task 1 | stop processing: n_tokens = 77, truncated = 0
"""
