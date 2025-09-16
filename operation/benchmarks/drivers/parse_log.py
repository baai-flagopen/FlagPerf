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
    
    if os.path.isfile(save_log_path):
        with open(save_log_path, 'r+', encoding='utf-8') as file_r:
            try:
                file_r_json = file_r.read()
                res = json.loads(file_r_json)
                # 处理性能测试日志
                result_data = get_result_data(performance_log_file, res, spectflops, mode, warmup)
                # 处理正确性测试日志
                result_data = get_correctness_data(correctness_log_file, result_data)
                file_r.seek(0)
                file_r.write(json.dumps(result_data, ensure_ascii=False))
                file_r.truncate()
            except json.decoder.JSONDecodeError:
                print("JSONDecodeError json file content is None")
    else:
        with open(save_log_path, 'w') as file_w:
            res = defaultdict(dict)
            # 处理性能测试日志
            result_data = get_result_data(performance_log_file, res, spectflops, mode, warmup)
            # 处理正确性测试日志
            result_data = get_correctness_data(correctness_log_file, result_data)
            file_w.write(json.dumps(result_data, ensure_ascii=False))


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
    current_test = None
    test_results = []
    
    for line in lines:
        line = line.strip()
        
        # 查找测试开始的标志
        if "test_accuracy_" in line and "::" in line:
            # 提取算子名称，例如：test_blas_ops.py::test_accuracy_mm
            parts = line.split("::")
            if len(parts) >= 2:
                test_name = parts[1]
                if "test_accuracy_" in test_name:
                    current_test = test_name.replace("test_accuracy_", "")
        
        # 查找测试结果
        elif "PASSED" in line:
            if current_test:
                correctness_info[current_test] = {
                    "correctness_status": "passed",
                    "correctness_details": "Test passed successfully"
                }
                test_results.append(f"{current_test}: PASSED")
                current_test = None
                
        elif "FAILED" in line:
            if current_test:
                correctness_info[current_test] = {
                    "correctness_status": "failed", 
                    "correctness_details": "Test failed"
                }
                test_results.append(f"{current_test}: FAILED")
                current_test = None
                
        elif "ERROR" in line:
            if current_test:
                correctness_info[current_test] = {
                    "correctness_status": "error",
                    "correctness_details": "Test encountered an error"
                }
                test_results.append(f"{current_test}: ERROR")
                current_test = None
    
    # 如果没有找到具体的测试结果，尝试从整体结果中推断
    if not correctness_info:
        if "failed" in log_content.lower():
            correctness_info["general"] = {
                "correctness_status": "failed",
                "correctness_details": "General test failure detected"
            }
        elif "passed" in log_content.lower() or "ok" in log_content.lower():
            correctness_info["general"] = {
                "correctness_status": "passed", 
                "correctness_details": "General test success detected"
            }
        else:
            correctness_info["general"] = {
                "correctness_status": "unknown",
                "correctness_details": "Unable to determine test result"
            }
    
    return correctness_info