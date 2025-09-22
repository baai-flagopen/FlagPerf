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
from collections import defaultdict
import ast

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
OP_PATH = os.path.abspath(os.path.join(CURR_PATH, "../"))
from formatMDfile import *


def extract_arrays_from_shape_detail(shape_detail):
    """
    从shape_detail中提取所有的数组，忽略其他类型的数据
    返回：(arrays_only_string, formatted_display_string)
    """
    try:
        # 如果shape_detail是字符串，尝试解析为Python对象
        if isinstance(shape_detail, str):
            try:
                parsed_data = ast.literal_eval(shape_detail)
            except:
                # 如果解析失败，直接返回原字符串
                return shape_detail, shape_detail
        else:
            parsed_data = shape_detail
        
        # 提取所有的数组（列表）
        arrays = []
        
        def extract_arrays_recursive(data):
            """递归提取所有数组"""
            if isinstance(data, list):
                # 检查这个列表是否是数值数组
                if all(isinstance(x, (int, float)) for x in data) and len(data) > 0:
                    arrays.append(data)
                else:
                    # 如果不是纯数值数组，递归检查子元素
                    for item in data:
                        extract_arrays_recursive(item)
        
        extract_arrays_recursive(parsed_data)
        
        if arrays:
            # 用于匹配的字符串（只包含数组）
            arrays_only = str(arrays)
            # 用于显示的字符串（格式化的数组）
            display_arrays = []
            for arr in arrays:
                if isinstance(arr, list):
                    display_arrays.append(f"[{', '.join(map(str, arr))}]")
            display_string = ", ".join(display_arrays)
            return arrays_only, display_string
        else:
            # 如果没有找到数组，返回原始值
            return str(shape_detail), str(shape_detail)
            
    except Exception as e:
        print(f"Error parsing shape_detail {shape_detail}: {e}")
        return str(shape_detail), str(shape_detail)


def find_valid_timestamp_dirs(result_base_dir, max_count=3):
    """
    查找包含有效result.json的时间戳目录
    返回最多max_count个有效的时间戳目录路径
    """
    timestamp_dirs = []
    if os.path.exists(result_base_dir):
        for item in os.listdir(result_base_dir):
            item_path = os.path.join(result_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("run"):
                timestamp_dirs.append(item)
    
    if not timestamp_dirs:
        print("No timestamp directories found in result folder")
        return []
    
    # 按时间戳排序，最新的在最后
    sorted_timestamp_dirs = sorted(timestamp_dirs)
    valid_dirs = []
    
    # 从最新的开始检查，向前查找
    for timestamp_dir in reversed(sorted_timestamp_dirs):
        if len(valid_dirs) >= max_count:
            break
            
        timestamp_path = os.path.join(result_base_dir, timestamp_dir)
        result_json_path = os.path.join(timestamp_path, "result.json")
        
        # 检查result.json是否存在且不为空
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 文件不为空
                        json_data = json.loads(content)
                        if json_data:  # JSON内容不为空
                            valid_dirs.append(timestamp_path)
                            print(f"Found valid result.json in: {timestamp_path}")
                        else:
                            print(f"Skipping empty JSON in: {timestamp_path}")
                    else:
                        print(f"Skipping empty file: {result_json_path}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"Skipping invalid JSON in {timestamp_path}: {e}")
        else:
            print(f"Skipping directory without result.json: {timestamp_path}")
    
    return valid_dirs


def merge_result_json_files(valid_dirs):
    """
    合并多个result.json文件，根据op_name、dtype、shape_detail三个字段进行匹配
    只有这三个字段完全一样的数据才会合并
    """
    merged_data = defaultdict(dict)
    
    for dir_path in valid_dirs:
        result_json_path = os.path.join(dir_path, "result.json")
        print(f"Processing: {result_json_path}")
        
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                json_data = json.loads(f.read())
                
                for key, value in json_data.items():
                    # 提取三个关键字段作为匹配条件
                    op_name = value.get("op_name", "")
                    dtype = value.get("dtype", "")
                    shape_detail_raw = value.get("shape_detail", "")
                    
                    # 从shape_detail中提取数组部分用于匹配
                    arrays_only, display_shape = extract_arrays_from_shape_detail(shape_detail_raw)
                    
                    # 创建唯一标识符：op_name_dtype_arrays_only
                    # 只有算子名、数据类型、数组形状完全一样的数据才会合并
                    unique_key = f"{op_name}_{dtype}_{arrays_only}"
                    
                    # 合并数据到unique_key下
                    if unique_key not in merged_data:
                        merged_data[unique_key] = {}
                        print(f"  Creating new entry for: {op_name}_{dtype}_{display_shape}")
                    else:
                        print(f"  Merging data into existing entry: {op_name}_{dtype}_{display_shape}")
                    
                    # 更新所有字段（相同unique_key的数据会合并字段）
                    merged_data[unique_key].update(value)
                    
                    # 更新shape_detail为只包含数组的显示格式
                    merged_data[unique_key]["shape_detail"] = display_shape
                    
        except Exception as e:
            print(f"Error processing {result_json_path}: {e}")
    
    return dict(merged_data)


def find_correctness_log_dir(valid_dirs):
    """
    在有效目录中查找包含正确性日志的目录
    优先查找merged_correctness.log.txt，其次查找传统的correctness.log.txt
    """
    for dir_path in valid_dirs:
        # 优先查找合并的正确性日志
        merged_correctness_log_path = os.path.join(dir_path, "merged_correctness.log.txt")
        if os.path.exists(merged_correctness_log_path):
            try:
                # 检查文件是否为空
                with open(merged_correctness_log_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 文件不为空
                        print(f"Found valid merged_correctness.log.txt in: {dir_path}")
                        return dir_path, "merged_correctness.log.txt"
                    else:
                        print(f"Skipping empty merged_correctness.log.txt in: {dir_path}")
            except Exception as e:
                print(f"Error reading merged_correctness.log.txt in {dir_path}: {e}")
        
        # 如果没有找到合并日志，查找传统的正确性日志
        correctness_log_path = os.path.join(dir_path, "correctness.log.txt")
        if os.path.exists(correctness_log_path):
            try:
                # 检查文件是否为空
                with open(correctness_log_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 文件不为空
                        print(f"Found valid correctness.log.txt in: {dir_path}")
                        return dir_path, "correctness.log.txt"
                    else:
                        print(f"Skipping empty correctness.log.txt in: {dir_path}")
            except Exception as e:
                print(f"Error reading correctness.log.txt in {dir_path}: {e}")
        else:
            print(f"No correctness log found in: {dir_path}")
    
    print("Warning: No valid correctness log found in any directory")
    return None, None


def main(vendor, shm_size, chip):
    result_base_dir = os.path.join(OP_PATH, "result")
    
    # 查找最多3个有效的时间戳目录
    valid_dirs = find_valid_timestamp_dirs(result_base_dir, max_count=3)
    
    if not valid_dirs:
        print("No valid timestamp directories with result.json found")
        return
    
    print(f"Found {len(valid_dirs)} valid directories for merging")
    
    # 合并所有result.json文件
    merged_data = merge_result_json_files(valid_dirs)
    
    if not merged_data:
        print("No data to merge")
        return
    
    print(f"Merged data contains {len(merged_data)} unique entries")
    
    # 将合并后的数据保存到result目录下（不在时间戳目录内）
    merged_json_path = os.path.join(result_base_dir, "merged_result.json")
    with open(merged_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"Saved merged data to: {merged_json_path}")
    
    # 生成最终的MD文件到result目录下（覆盖上一次结果）
    readme_output_dir = result_base_dir
    
    # 查找包含正确性日志的目录
    correctness_source_dir, correctness_log_filename = find_correctness_log_dir(valid_dirs)
    if correctness_source_dir is None:
        correctness_source_dir = readme_output_dir
        correctness_log_filename = "correctness.log.txt"  # 默认文件名
        print("Warning: No correctness log found in any valid directories")
    
    render(merged_data, correctness_source_dir, vendor, shm_size, chip, correctness_log_filename)
    
    # 将生成的README.md移动到result目录下
    source_readme = os.path.join(correctness_source_dir, "README.md")
    target_readme = os.path.join(readme_output_dir, "README.md")
    if source_readme != target_readme and os.path.exists(source_readme):
        import shutil
        shutil.move(source_readme, target_readme)
        print(f"Moved README.md to: {target_readme}")
    else:
        print(f"Generated final README.md in: {target_readme}")




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