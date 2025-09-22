# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import os
from collections import defaultdict
from loguru import logger


def parse_log_file(spectflops, mode, warmup, log_dir, result_log_path):
    performance_log_file = os.path.join(log_dir, "result.log.txt")
    correctness_log_file = os.path.join(log_dir, "correctness.log.txt")
    save_log_path = os.path.join(result_log_path, "result.json")
    
    # 处理性能测试日志（合并而不是覆盖）
    if os.path.isfile(save_log_path):
        with open(save_log_path, 'r+', encoding='utf-8') as file_r:
            try:
                file_r_json = file_r.read()
                if file_r_json.strip():  # 检查文件内容不为空
                    existing_data = json.loads(file_r_json)
                else:
                    print("JSON file is empty, initializing new data")
                    existing_data = {}
                
                # 处理当前算子的性能测试日志（保持原有逻辑兼容性）
                # 将现有数据传入get_result_data，让它在现有基础上添加新数据
                result_data = get_result_data(performance_log_file, existing_data, spectflops, mode, warmup)
                
                # 注意：暂时不在这里处理正确性数据，留给后续的合并函数处理
                # 这样避免了正确性数据被覆盖的问题
                
                file_r.seek(0)
                file_r.write(json.dumps(result_data, ensure_ascii=False, indent=2))
                file_r.truncate()
                
            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}, reinitializing data")
                # JSON解析失败时，重新处理数据
                new_data = get_result_data(performance_log_file, {}, spectflops, mode, warmup)
                file_r.seek(0)
                file_r.write(json.dumps(new_data, ensure_ascii=False, indent=2))
                file_r.truncate()
    else:
        # 首次创建文件，只包含性能数据
        performance_data = get_result_data(performance_log_file, {}, spectflops, mode, warmup)
        with open(save_log_path, 'w') as file_w:
            file_w.write(json.dumps(performance_data, ensure_ascii=False, indent=2))
    
    print(f"Performance data saved to: {save_log_path}")
    print(f"Correctness log available at: {correctness_log_file}")


""" 参数说明
# 时延：1 无预热时延 Latency-No warmup：no_warmup_latency，2 预热时延 Latency-Warmup：warmup_latency
# 吞吐率：3 Raw-Throughput原始吞吐：raw_throughput， 4 Core-Throughput是核心吞吐：core_throughput
# 算力：5 实际算力开销：ctflops， 6 实际算力利用率：cfu， 7 实际算力开销-内核时间：ktflops， 8 实际算力利用率-内核时间：kfu
"""
def get_result_data(log_file, res, spectflops, mode, warmup):
    with open(log_file, 'r') as file_r:
        lines = file_r.readlines()
        for line in lines:
            if line.startswith("[INFO]"):
                json_data = line[6:].strip()
                try:
                    data = json.loads(json_data)
                    op_name = data.get("op_name")
                    dtype = data.get("dtype")
                    results = data.get("result")
                    for result in results:
                        shape_detail = result.get("shape_detail")
                        latency_base = result.get("latency_base")
                        if mode == "cpu" and warmup == "0":
                            no_warmup_latency = result.get("latency")
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_cpu_nowarm": latency_base,
                                "no_warmup_latency": no_warmup_latency
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "operator" and warmup == "0":
                            # 处理operator模式无预热情况，类似于cpu模式
                            no_warmup_latency = result.get("latency")
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_operator_nowarm": latency_base,
                                "no_warmup_latency": no_warmup_latency
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cpu" and warmup != "0":
                            warmup_latency = result.get("latency")
                            raw_throughput = 1 / float(warmup_latency)
                            ctflops = result.get("tflops")
                            if ctflops is None:
                                cfu = None
                            else:
                                cfu = round(100.0 * float(ctflops) / 1e12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_cpu_warm": latency_base,
                                "warmup_latency": warmup_latency,
                                "raw_throughput": raw_throughput,
                                "ctflops": ctflops,
                                "cfu": cfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "cuda" and warmup != "0":
                            kerneltime = result.get("latency")
                            core_throughput = 1 / float(kerneltime)
                            ktflops = result.get("tflops")
                            if ktflops is None:
                                kfu = None
                            else:
                                kfu = round(100.0 * float(ktflops) / 1E12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_cuda_warm": latency_base,
                                "kerneltime": kerneltime,
                                "core_throughput": core_throughput,
                                "ktflops": ktflops,
                                "kfu": kfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "operator" and warmup != "0":
                            # 处理operator模式，类似于cpu模式
                            warmup_latency = result.get("latency")
                            raw_throughput = 1 / float(warmup_latency)
                            ctflops = result.get("tflops")
                            if ctflops is None:
                                cfu = None
                            else:
                                cfu = round(100.0 * float(ctflops) / 1e12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_operator_warm": latency_base,
                                "warmup_latency": warmup_latency,
                                "raw_throughput": raw_throughput,
                                "ctflops": ctflops,
                                "cfu": cfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                        elif mode == "kernel" and warmup != "0":
                            # 处理kernel模式，类似于cuda模式
                            kerneltime = result.get("latency")
                            core_throughput = 1 / float(kerneltime)
                            ktflops = result.get("tflops")
                            if ktflops is None:
                                kfu = None
                            else:
                                kfu = round(100.0 * float(ktflops) / 1E12 / float(spectflops), 2)
                            parse_data = {
                                "op_name": op_name,
                                "dtype": dtype,
                                "shape_detail": shape_detail,
                                "latency_base_kernel_warm": latency_base,
                                "kerneltime": kerneltime,
                                "core_throughput": core_throughput,
                                "ktflops": ktflops,
                                "kfu": kfu
                            }
                            res[f"{op_name}_{dtype}_{shape_detail}"].update(parse_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
        return res


def get_correctness_data(correctness_log_file, result_data):
    """
    解析正确性测试日志，提取测试结果并整合到结果数据中
    """
    if not os.path.exists(correctness_log_file):
        logger.warning(f"Correctness log file not found: {correctness_log_file}")
        # 为所有已有的测试结果添加默认的正确性状态
        for key in result_data:
            result_data[key]["correctness_status"] = "not_tested"
            result_data[key]["correctness_details"] = "Correctness log file not found"
        return result_data
    
    try:
        with open(correctness_log_file, 'r', encoding='utf-8') as file_r:
            log_content = file_r.read()
            
            # 首先尝试解析JSON格式
            correctness_info = parse_pytest_json_output(log_content)
            
            # 如果JSON解析失败，尝试解析文本格式
            if not correctness_info:
                correctness_info = parse_pytest_text_output(log_content)
            
            # 将正确性信息整合到结果数据中
            for key in result_data:
                op_name = result_data[key].get("op_name", "")
                dtype = result_data[key].get("dtype", "")
                
                # 尝试多种匹配方式
                matched = False
                for test_key, test_info in correctness_info.items():
                    # 检查算子名称是否匹配
                    if op_name in test_key.lower():
                        result_data[key].update(test_info)
                        matched = True
                        break
                
                if not matched:
                    # 如果没有找到对应算子的正确性测试结果，设置默认值
                    result_data[key]["correctness_status"] = "not_found"
                    result_data[key]["correctness_details"] = f"No correctness test found for {op_name}"
                    
    except Exception as e:
        logger.error(f"Error reading correctness log file: {e}")
        # 出错时为所有结果添加错误状态
        for key in result_data:
            result_data[key]["correctness_status"] = "error"
            result_data[key]["correctness_details"] = f"Error reading correctness log: {str(e)}"
    
    return result_data


def parse_pytest_json_output(log_content):
    """
    解析pytest JSON格式输出，提取测试结果
    """
    correctness_info = {}
    
    try:
        # 尝试解析JSON格式
        json_data = json.loads(log_content)
        
        for test_key, test_data in json_data.items():
            # 从测试键中提取算子名称
            # 例如：tests/test_attention_ops.py::test_sdpa_legacy[dtype0-False-4-8-8-1024-1024-64-False]
            op_name = extract_op_name_from_test_key(test_key)
            
            # 获取测试结果
            result = test_data.get("result")
            params = test_data.get("params", {})
            
            # 确定测试状态
            if result is None:
                status = "not_run"
                details = "Test was not executed"
            elif result == "passed" or result is True:
                status = "passed"
                details = "Test passed successfully"
            elif result == "failed" or result is False:
                status = "failed"
                details = "Test failed"
            elif result == "error":
                status = "error"
                details = "Test encountered an error"
            else:
                status = "unknown"
                details = f"Unknown test result: {result}"
            
            correctness_info[test_key] = {
                "correctness_status": status,
                "correctness_details": details,
                "test_params": params,
                "op_name": op_name
            }
            
    except json.JSONDecodeError:
        # 如果不是JSON格式，返回空字典
        return {}
    except Exception as e:
        logger.error(f"Error parsing pytest JSON output: {e}")
        return {}
    
    return correctness_info


def extract_op_name_from_test_key(test_key):
    """
    从测试键中提取算子名称
    例如：tests/test_attention_ops.py::test_sdpa_legacy -> sdpa
    """
    try:
        # 提取测试函数名
        if "::" in test_key:
            test_func = test_key.split("::")[-1]
            # 移除参数部分
            if "[" in test_func:
                test_func = test_func.split("[")[0]
            
            # 提取算子名称
            if "test_" in test_func:
                # 移除test_前缀
                op_part = test_func.replace("test_", "")
                # 移除常见后缀
                for suffix in ["_legacy", "_accuracy", "_correctness", "_ops"]:
                    op_part = op_part.replace(suffix, "")
                return op_part
                
        return "unknown"
    except Exception:
        return "unknown"


def parse_pytest_text_output(log_content):
    """
    解析pytest文本格式输出，提取测试结果
    """
    correctness_info = {}
    
    lines = log_content.split('\n')
    passed_count = 0
    failed_count = 0
    error_count = 0
    
    for line in lines:
        line = line.strip()
        
        # 查找具体测试结果行，格式：test_blas_ops.py::test_accuracy_mm[dtype0-15-15-15] PASSED [1%]
        if "test_accuracy_" in line and "::" in line and (" PASSED " in line or " FAILED " in line or " ERROR " in line):
            # 提取算子名称
            parts = line.split("::")
            if len(parts) >= 2:
                test_func = parts[1].split()[0]  # 去除后面的PASSED等状态
                if "test_accuracy_" in test_func:
                    # 提取算子名称，例如从 test_accuracy_mm[dtype0-15-15-15] 中提取 mm
                    op_name = test_func.replace("test_accuracy_", "").split("[")[0]
                    
                    if " PASSED " in line:
                        passed_count += 1
                        status = "passed"
                        details = "Test passed successfully"
                    elif " FAILED " in line:
                        failed_count += 1
                        status = "failed"
                        details = "Test failed"
                    elif " ERROR " in line:
                        error_count += 1
                        status = "error"
                        details = "Test encountered an error"
                    
                    # 为每个算子保存状态（如果已存在，保持最严重的状态）
                    if op_name not in correctness_info:
                        correctness_info[op_name] = {
                            "correctness_status": status,
                            "correctness_details": details,
                            "test_count": 1
                        }
                    else:
                        # 更新计数
                        correctness_info[op_name]["test_count"] += 1
                        # 如果有失败，更新状态
                        if status == "failed" or status == "error":
                            correctness_info[op_name]["correctness_status"] = status
                            correctness_info[op_name]["correctness_details"] = details
    
    # 查找总结行，例如：81 passed, 1431 deselected, 1 warning
    for line in lines:
        if " passed" in line and ("deselected" in line or "warning" in line):
            # 解析总结信息
            summary_info = line.strip()
            break
    else:
        summary_info = f"{passed_count} passed, {failed_count} failed, {error_count} errors"
    
    # 如果没有找到具体的测试结果，尝试从整体结果中推断
    if not correctness_info:
        if failed_count > 0 or error_count > 0:
            correctness_info["general"] = {
                "correctness_status": "failed",
                "correctness_details": f"Tests failed: {summary_info}"
            }
        elif passed_count > 0:
            correctness_info["general"] = {
                "correctness_status": "passed", 
                "correctness_details": f"Tests passed: {summary_info}"
            }
        else:
            correctness_info["general"] = {
                "correctness_status": "unknown",
                "correctness_details": "Unable to determine test result"
            }
    else:
        # 为所有算子添加总结信息
        for op_name in correctness_info:
            correctness_info[op_name]["test_summary"] = summary_info
    
    return correctness_info


def merge_correctness_logs_from_subdirs(timestamp_dir):
    """
    合并时间戳目录下所有算子子目录的正确性日志
    生成合并的正确性日志文件和更新的result.json
    """
    print(f"=== Starting correctness logs merge from: {timestamp_dir} ===")
    
    merged_correctness = {}
    merged_log_content = []
    processed_cases = []
    
    # 遍历所有算子子目录
    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if not os.path.isdir(item_path) or not item.startswith("opv2:"):
            continue
            
        print(f"Processing case directory: {item}")
        
        # 查找算子子目录中的正确性日志
        case_correctness_found = False
        for subdir in os.listdir(item_path):
            subdir_path = os.path.join(item_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            correctness_file = os.path.join(subdir_path, "correctness.log.txt")
            
            if os.path.exists(correctness_file) and os.path.getsize(correctness_file) > 0:
                print(f"  Found correctness log: {correctness_file}")
                case_correctness_found = True
                
                try:
                    with open(correctness_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    # 解析正确性信息
                    correctness_info = parse_pytest_json_output(log_content)
                    if not correctness_info:
                        correctness_info = parse_pytest_text_output(log_content)
                    
                    # 合并到总的正确性数据中
                    for test_key, test_data in correctness_info.items():
                        # 为测试键添加case前缀，避免重复
                        merged_key = f"{item}::{test_key}"
                        merged_correctness[merged_key] = test_data
                    
                    # 添加日志内容（用于生成合并的日志文件）
                    merged_log_content.append(f"\n{'='*60}\n")
                    merged_log_content.append(f"CASE: {item}\n")
                    merged_log_content.append(f"SOURCE: {correctness_file}\n")
                    merged_log_content.append(f"{'='*60}\n")
                    merged_log_content.append(log_content)
                    merged_log_content.append(f"\n{'='*60}\n")
                    
                    processed_cases.append(item)
                    print(f"  Successfully processed correctness data for {item}")
                    
                except Exception as e:
                    print(f"  Error processing {correctness_file}: {e}")
        
        if not case_correctness_found:
            print(f"  No correctness log found for case: {item}")
    
    # 生成合并的正确性日志文件
    merged_log_path = os.path.join(timestamp_dir, "merged_correctness.log.txt")
    try:
        with open(merged_log_path, 'w', encoding='utf-8') as f:
            f.write(f"MERGED CORRECTNESS LOG\n")
            f.write(f"Generated from {len(processed_cases)} cases: {', '.join(processed_cases)}\n")
            f.write(f"Timestamp: {timestamp_dir.split('/')[-1]}\n")
            f.write("".join(merged_log_content))
        print(f"Merged correctness log saved to: {merged_log_path}")
    except Exception as e:
        print(f"Error creating merged correctness log: {e}")
        merged_log_path = None
    
    print(f"=== Correctness merge completed. Processed {len(processed_cases)} cases ===")
    return merged_correctness, merged_log_path, processed_cases


def finalize_correctness_logs_only(timestamp_dir):
    """
    只合并正确性日志文件，不修改result.json
    result.json只包含性能数据，正确性数据由MD格式化脚本单独处理
    """
    print(f"=== Merging correctness logs from: {timestamp_dir} ===")
    
    # 只合并正确性日志，不处理result.json
    merged_correctness, merged_log_path, processed_cases = merge_correctness_logs_from_subdirs(timestamp_dir)
    
    # 创建汇总信息
    summary = {
        "timestamp_dir": timestamp_dir,
        "processed_cases": processed_cases,
        "merged_log_path": merged_log_path,
        "total_correctness_entries": len(merged_correctness)
    }
    
    print(f"=== Correctness logs merge completed ===")
    print(f"Processed {len(processed_cases)} cases")
    print(f"Generated merged log: {merged_log_path}")
    return summary