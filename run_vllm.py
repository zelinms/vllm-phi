from vllm import LLM, SamplingParams

# from huggingface_hub import login
# login(token="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")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is a",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
llm = LLM(model="/home/phi3-moe-RC-3-0-16",
          trust_remote_code=True,
          quantization="fp8",
          dtype="bfloat16", tensor_parallel_size=1, max_model_len=200)
# llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=2, max_model_len=200)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# print (outputs)
# import pdb;pdb.set_trace()
for output in outputs:
    prompt = output.prompt
    generated_text = outputs[0].outputs[0].text
    print (prompt)
    print (generated_text)
    print ('-------------')
    
    