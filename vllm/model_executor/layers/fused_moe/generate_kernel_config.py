import os
import json
from multiprocessing import Process
import multiprocessing
import subprocess
import torch

config_bucket = dict()  # expert,intermediate_size : { "tokens": config}
config_best_time = dict()
device_name = torch.cuda.get_device_name().replace(" ", "_")


def read_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)

                    if data["time"] < 0.01:
                        continue

                    kstr = f"E={data['expert_num']},N={data['intermediate_size']},device_name={device_name}.json"

                    if kstr not in config_bucket:
                        config_bucket[kstr] = dict()

                    if data["tokens"] not in config_bucket[kstr]:
                        config_best_time[kstr + str(data["tokens"])] = data["time"]
                        config_bucket[kstr][data["tokens"]] = data["config"]
                    else:
                        if data["time"] < config_best_time[kstr + str(data["tokens"])]:
                            config_bucket[kstr][data["tokens"]] = data["config"]
                            config_best_time[kstr + str(data["tokens"])] = data["time"]

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")


read_json_files("cache")


if not os.path.exists("new_configs"):
    os.mkdir("new_configs")

for fname in config_bucket:
    with open(f"new_configs/{fname}", "w") as file:
        json.dump(config_bucket[fname], file, indent=4)

print(config_best_time)
