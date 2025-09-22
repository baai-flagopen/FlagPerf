 # Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import os
import time
from argparse import ArgumentParser, Namespace
import yaml
import sys
import subprocess

sys.path.append("..")
from drivers.utils import *
from drivers.calculate import *
from drivers.parse_log import *


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")
    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="op name like mm")
    parser.add_argument("--spectflops",
                        type=str,
                        required=True,
                        help="spectflops of current dataformat")

    parser.add_argument("--dataformat",
                        type=str,
                        required=True,
                        help="like FP32,FP16")

    parser.add_argument("--oplib",
                        type=str,
                        required=True,
                        help="impl like pytorch/flaggems/cpp")

    parser.add_argument("--chip",
                        type=str,
                        required=True,
                        help="chip like A100_40_SXM")

    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        help="mode like cpu")

    parser.add_argument("--warmup",
                        type=str,
                        required=True,
                        help="warmup")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs log dir")

    parser.add_argument("--result_log_path",
                        type=str,
                        required=True,
                        help="result log path for FlagPerf/operation/result")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def main(config):
    print("=== Starting main function ===")
    print(f"Config: {config}")
    print(f"Case name: {config.case_name}")
    print(f"Log dir: {config.log_dir}")
    print(f"Mode: {config.mode}")
    print(f"Warmup: {config.warmup}")

    try:
        print("=== Starting correctness test ===")
        print(f"FLAGGEMS_WORK_DIR environment: {os.environ.get('FLAGGEMS_WORK_DIR', 'NOT_SET')}")
        
        # 找到container_main.py已经创建的目录，而不是创建新目录
        # container_main.py第102行：logfile = os.path.join(config.log_dir, config.case_name, config.host_addr + "_noderank" + str(config.node_rank), "container_main.log.txt")
        # 注意：container_main.py中的config.case_name是完整名称，config.host_addr可能是"localhost"
        
        # 重新构造完整的算子名称
        full_case_name = f"opv2:{config.case_name}:{config.dataformat}:{config.spectflops}:{config.oplib}:{config.chip}"
        algorithm_dir = os.path.join(config.log_dir, full_case_name)
        
        # 查找已存在的noderank目录（container_main.py已经创建的）
        case_log_dir = None
        if os.path.exists(algorithm_dir):
            print(f"Searching for existing noderank directory in: {algorithm_dir}")
            for item in os.listdir(algorithm_dir):
                item_path = os.path.join(algorithm_dir, item)
                if os.path.isdir(item_path) and "noderank0" in item:
                    case_log_dir = item_path
                    print(f"Found existing noderank directory: {case_log_dir}")
                    break
        
        # 如果找不到现有目录，使用localhost作为默认（通常container_main.py使用localhost）
        if case_log_dir is None:
            case_log_dir = os.path.join(algorithm_dir, "localhost_noderank0")
            os.makedirs(case_log_dir, exist_ok=True)
            print(f"Created default case log directory: {case_log_dir}")
        
        print(f"Full case name: {full_case_name}")
        print(f"Using case log directory: {case_log_dir}")
        print(f"Expected operation.log.txt location: {os.path.join(case_log_dir, 'operation.log.txt')}")
        
        correctness = do_correctness(config.case_name, case_log_dir)
        print(f"do_correctness returned: {correctness}")
        correctness = correctness == 0
        print(f"Correctness result: {correctness}")

        print("=== Starting performance test ===")
        # 创建性能测试开始标记文件
        perf_start_marker = os.path.join(case_log_dir, "performance_started.marker")
        with open(perf_start_marker, 'w') as f:
            f.write(f"Performance test started at {time.time()}\n")
        print(f"Created performance start marker: {perf_start_marker}")
        
        # test operation performance
        print(f"Calling do_performance with mode={config.mode}, warmup={config.warmup}, log_dir={case_log_dir}")
        performance = do_performance(config.mode, config.warmup, case_log_dir)
        print(f"do_performance returned: {performance}")
        
        # 创建性能测试完成标记文件
        perf_end_marker = os.path.join(case_log_dir, "performance_completed.marker")
        with open(perf_end_marker, 'w') as f:
            f.write(f"Performance test completed at {time.time()}\n")
        print(f"Created performance end marker: {perf_end_marker}")

        # Check if performance is a tuple (success case) or single value (error case)
        if isinstance(performance, tuple):
            performance_success = performance[0] == 0 and performance[1] == 0
            print(f"Performance tuple result: {performance}, success: {performance_success}")
        else:
            performance_success = performance == 0
            print(f"Performance single result: {performance}, success: {performance_success}")

        print("=== Starting log parsing ===")
        print(f"Calling parse_log_file with spectflops={config.spectflops}, mode={config.mode}, warmup={config.warmup}")
        parse_log_file(config.spectflops, config.mode, config.warmup, case_log_dir, config.result_log_path)
        print("=== Log parsing completed ===")
        print("=== Main function completed successfully ===")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        print("=== Main function failed ===")
        
    print("=== Exiting main function ===")

    # dtype = {
    #     "FP32": torch.float32,
    #     "FP16": torch.float16,
    #     "BF16": torch.bfloat16,
    #     "INT32": torch.int32,
    #     "INT16": torch.int16,
    #     "BOOL": torch.bool
    #     }
    # set_ieee_float32(config.vendor)
    #
    #
    # m = case_config.Melements
    #
    #
    # a = torch.randn(m, 1024, 1024, dtype=dtype[config.dataformat]).to(0)
    #
    # latency_nowarm, latency_warm, cputime, kerneltime = do_test(
    #     torch.abs, (a, ), host_device_sync, config, case_config)
    #
    # op2flops = lambda x: x * m * 1024 * 1024
    #
    # perf_result = cal_perf(cputime, kerneltime, op2flops,
    #                        config.spectflops)
    # print_result(config, config.case_name, *perf_result, correctness,
    #              latency_nowarm, latency_warm)

if __name__ == "__main__":
    print("=== Starting opv2/main.py ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Command line arguments: {sys.argv}")
    
    try:
        config = parse_args()
        print("=== Arguments parsed successfully ===")
        print(f"Arguments received: {config}")
        
        if config.oplib == "flaggems":
            import flag_gems
            flag_gems.enable()
            print("Using flaggems")
        else:
            print("Using nativetorch")
        
        print("=== Calling main function ===")
        main(config)
        print("=== Program completed successfully ===")
        
    except Exception as e:
        print(f"=== ERROR in main execution: {e} ===")
        import traceback
        traceback.print_exc()
        print("=== Program failed ===")
        sys.exit(1)