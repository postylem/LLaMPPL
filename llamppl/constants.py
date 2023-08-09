from .llama_cpp import llama_token_eos, llama_token_bos

EOS = llama_token_eos()
BOS = llama_token_bos()

WEIGHTS_PATH = "vendor/llama.cpp/models/7B/ggml-model-q4_0.bin"