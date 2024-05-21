import os
import json
from multiprocessing import Process
import multiprocessing
import subprocess
import torch

BLOCK_SIZE_M = set()
BLOCK_SIZE_N = set()
BLOCK_SIZE_K = set()
GROUP_SIZE_M = set()
num_warps = set()
num_stages = set()


def read_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)
                    for key, value in data.items():
                        BLOCK_SIZE_M.add(value.get("BLOCK_SIZE_M"))
                        BLOCK_SIZE_N.add(value.get("BLOCK_SIZE_N"))
                        BLOCK_SIZE_K.add(value.get("BLOCK_SIZE_K"))
                        GROUP_SIZE_M.add(value.get("GROUP_SIZE_M"))
                        num_warps.add(value.get("num_warps"))
                        num_stages.add(value.get("num_stages"))

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")


read_json_files("configs")

print("BLOCK_SIZE_M:", BLOCK_SIZE_M)
print("BLOCK_SIZE_N:", BLOCK_SIZE_N)
print("BLOCK_SIZE_K:", BLOCK_SIZE_K)
print("GROUP_SIZE_M:", GROUP_SIZE_M)
print("num_warps:", num_warps)
print("num_stages:", num_stages)


def invoke_profiling(
    tokens=1024,
    intermediate_size=6400,
    expert_num=16,
    m=64,
    n=64,
    k=32,
    group_m=8,
    warps=4,
    stages=2,
):
    if warps is None:
        warps = 4
    if stages is None:
        stages = 2

    device_name = torch.cuda.get_device_name().replace(" ", "+")
    args = {
        "config": {
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": n,
            "BLOCK_SIZE_K": k,
            "GROUP_SIZE_M": group_m,
            "num_warps": warps,
            "num_stages": stages,
        },
        "tokens": tokens,
        "intermediate_size": intermediate_size,
        "expert_num": expert_num,
    }

    args_str = json.dumps(args)
    print("Passing Args:", args_str)

    jsonfname = f"cache/config_{m}_{n}_{k}_{group_m}_{warps}_{stages}_{tokens}_{intermediate_size}_{expert_num}_{device_name}.json"

    if not os.path.exists("cache"):
        os.mkdir("cache")

    if os.path.exists(jsonfname):
        with open(jsonfname, "r") as json_file:
            data = json.load(json_file)
            print("Cached:", data)
            return data["time"]

    try:
        result = subprocess.run(
            ["python", "profile_kernel.py"],
            input=args_str,
            capture_output=True,
            text=True,
            check=True,
        )
        this_time = float(result.stdout)
        print("Time:", this_time)
        args["time"] = this_time

        with open(jsonfname, "w") as json_file:
            print("Caching:", args)
            json.dump(args, json_file, indent=4)

    except subprocess.CalledProcessError as e:
        args["time"] = -1
        with open(jsonfname, "w") as json_file:
            print("Caching:", args)
            json.dump(args, json_file, indent=4)
        print("Error:", e)

    return args["time"]


def get_default_latency(tokens, intermediate_size, expert_num):
    if tokens > expert_num:
        return invoke_profiling(
            tokens=tokens, intermediate_size=intermediate_size, expert_num=expert_num
        )
    return invoke_profiling(
        tokens=tokens,
        intermediate_size=intermediate_size,
        expert_num=expert_num,
        m=16,
        n=32,
        k=64,
        group_m=8,
        warps=4,
        stages=2,
    )


# searchspace = [1, 2, 4, 8, 16] + list(range(0, 256, 32)) + list(range(256, 4097, 256))
searchspace = [1] + [32] + [128] + list(range(256, 4097, 256))
# searchspace = (256, 1024, 1024 * 2, 1024 * 4)
intermediate_size = int(os.environ.get("INTERMEDIATE_SIZE", 6400))
expert_num = 16


def profile_configspace(configspace, rank, world_size, tokens):

    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    sub_configspace = split_list(configspace, world_size)[rank]

    for config in sub_configspace:
        invoke_profiling(
            tokens,
            intermediate_size,
            expert_num,
            config[0],
            config[1],
            config[2],
            config[3],
            config[4],
            config[5],
        )


for tokens in searchspace:
    print(f"Tokens: {tokens}")

    configspace = [(16, 32, 16, 8, 4, 2), (64, 64, 32, 8, 4, 2)]

    for m in BLOCK_SIZE_M:
        for n in BLOCK_SIZE_N:
            for k in BLOCK_SIZE_K:
                for group_m in GROUP_SIZE_M:
                    for warps in num_warps:
                        for stages in num_stages:
                            configspace.append((m, n, k, group_m, warps, stages))

    def remove_elements(lst):
        import random
        random.seed(1234)
        num_to_remove = int(len(lst) * 0.75)
        indices_to_remove = random.sample(range(len(lst)), num_to_remove)
        new_lst = [item for idx, item in enumerate(lst) if idx not in indices_to_remove]
        return new_lst

    configspace = remove_elements(configspace)

    device_name = torch.cuda.get_device_name().replace(" ", "+")
    gpu_id = int(os.environ["RANK"])
    gpu_count = int(os.environ["WORLD_SIZE"])

    print(f"GPU: {gpu_id}")
    profile_configspace(configspace, gpu_id, gpu_count, tokens)

    print("Done tuning")

    min_time = -1
    best_json = None
    for cfg in configspace:
        jsonfname = f"cache/config_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}_{cfg[4]}_{cfg[5]}_{tokens}_{intermediate_size}_{expert_num}_{device_name}.json"

        if not os.path.exists(jsonfname):
            continue

        with open(jsonfname, "r") as file:
            data = json.load(file)

        if data["time"] < min_time or min_time == -1:
            min_time = data["time"]
            best_json = data

    print(best_json)
