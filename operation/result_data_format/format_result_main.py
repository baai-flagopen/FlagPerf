# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import os
import sys
import yaml
from argparse import Namespace

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
OP_PATH = os.path.abspath(os.path.join(CURR_PATH, "../"))
from formatMDfile import *


def main(vendor, shm_size, chip):
    # 查找最新的时间戳目录
    result_base_dir = os.path.join(OP_PATH, "result")
    
    # 获取所有时间戳目录
    timestamp_dirs = []
    if os.path.exists(result_base_dir):
        for item in os.listdir(result_base_dir):
            item_path = os.path.join(result_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("run"):
                timestamp_dirs.append(item)
    
    if not timestamp_dirs:
        print("No timestamp directories found in result folder")
        return
    
    # 按时间戳排序，获取最新的
    latest_timestamp_dir = sorted(timestamp_dirs)[-1]
    latest_result_dir = os.path.join(result_base_dir, latest_timestamp_dir)
    
    print(f"Using latest result directory: {latest_result_dir}")
    
    # result.json直接在时间戳目录中
    result_json_file_path = os.path.join(latest_result_dir, "result.json")
    
    if not os.path.exists(result_json_file_path):
        print(f"result.json not found in {latest_result_dir}")
        return
    
    print(f"Found result.json at: {result_json_file_path}")
    
    # README.md生成在时间戳目录中（与result.json同一目录）
    readme_output_dir = latest_result_dir
    
    # render_base(readme_output_dir, vendor, shm_size, chip)
    with open(result_json_file_path, 'r') as f:
        content = json.loads(f.read())
        render(content, readme_output_dir, vendor, shm_size, chip)




if __name__ == "__main__":
    config_path = os.path.join(OP_PATH, "configs/host.yaml")
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
        config = Namespace(**config_dict)
        cases = []
        for case in config.CASES:
            cases.append(case)
        vendor = config.VENDOR
        shm_size = config.SHM_SIZE
        for run_case in cases:
            case_name = run_case
        test_file, op, dataformat, spectflops, oplib, chip = case_name.split(":")
        main(vendor, shm_size, chip)
    print("successful !!!")