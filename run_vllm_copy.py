from vllm import LLM, SamplingParams

from huggingface_hub import login
login(token="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")

prompts = [
    "Hello, my name is",
     "<|user|> What is the biggest city in Antarctic?<|end|><|assistant|>"
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    "DeepSpeed is a",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            min_tokens=1,
            stop=["<|end|>", "<|endoftext|>"],
            include_stop_str_in_output=True,
        )
# llm = LLM(model="/home/data/transformers_cache/moe-test-phi/phi3-moe-RC-3-0-15",#longMoE-RC_4_80_40_1
#         trust_remote_code=True,
#         tensor_parallel_size=1,
#         #quantization="fp8",
#         # kv_cache_dtype="fp8" ,
#           dtype="bfloat16",max_model_len=4096)# tensor_parallel_size=2, 
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=2, max_model_len=200)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# print (outputs)
# import pdb;pdb.set_trace()
for output in outputs:
    prompt = output.prompt
    out= output.outputs[0].text
    print (prompt)
    print (out)
    print ('-------------')
    
