# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from loguru import logger
import os
import sys
from argparse import ArgumentParser
import subprocess


def parse_args():
    parser = ArgumentParser(description=" ")

    parser.add_argument("--case_name",
                        type=str,
                        required=True,
                        help="case name")

    parser.add_argument("--nnodes",
                        type=int,
                        required=True,
                        help="number of node")

    parser.add_argument("--nproc_per_node",
                        type=int,
                        required=True,
                        help="*pu per node")

    parser.add_argument("--log_dir",
                        type=str,
                        required=True,
                        help="abs log dir")

    parser.add_argument("--vendor",
                        type=str,
                        required=True,
                        help="vendor name like nvidia")

    parser.add_argument("--log_level",
                        type=str,
                        required=True,
                        help="log level")

    parser.add_argument("--master_port",
                        type=int,
                        required=True,
                        help="master port")

    parser.add_argument("--master_addr",
                        type=str,
                        required=True,
                        help="master ip")

    parser.add_argument("--host_addr", type=str, required=True, help="my ip")

    parser.add_argument("--node_rank", type=int, required=True, help="my rank")

    parser.add_argument("--perf_path",
                        type=str,
                        required=True,
                        help="abs path for FlagPerf/base")

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args
    return args


def write_pid_file(pid_file_path, pid_file):
    '''Write pid file for watching the process later.
       In each round of case, we will write the current pid in the same path.
    '''
    pid_file_path = os.path.join(pid_file_path, pid_file)
    if os.path.exists(pid_file_path):
        os.remove(pid_file_path)
    file_d = open(pid_file_path, "w")
    file_d.write("%s\n" % os.getpid())
    file_d.close()


if __name__ == "__main__":
    config = parse_args()

    # 处理case_name中的冒号，将其替换为安全的字符用于文件路径
    safe_case_name = config.case_name.replace(":", "_")
    
    logfile = os.path.join(
        config.log_dir, safe_case_name,
        config.host_addr + "_noderank" + str(config.node_rank),
        "container_main.log.txt")
    
    # 确保日志目录存在
    log_dir = os.path.dirname(logfile)
    os.makedirs(log_dir, exist_ok=True)
    
    # 打印调试信息
    print(f"Container main starting with log file: {logfile}")
    print(f"Log directory: {log_dir}")
    print(f"Log directory exists: {os.path.exists(log_dir)}")
    
    logger.remove()
    logger.add(logfile, level=config.log_level)
    logger.add(sys.stdout, level=config.log_level)

    logger.info("Container main started with config:")
    logger.info(config)
    
    try:
        write_pid_file(config.log_dir, "start_base_task.pid")
        logger.info("Success Writing PID file at " +
                    os.path.join(config.log_dir, "start_base_task.pid"))
    except Exception as e:
        logger.error(f"Failed to write PID file: {e}")

    try:
        op, dataformat, spectflops, oplib, chip = config.case_name.split(":")
        logger.info(f"Parsed case_name: op={op}, dataformat={dataformat}, spectflops={spectflops}, oplib={oplib}, chip={chip}")
    except Exception as e:
        logger.error(f"Failed to parse case_name '{config.case_name}': {e}")
        sys.exit(1)

    case_dir = os.path.join(config.perf_path, "benchmarks", op)
    logger.info(f"Case directory: {case_dir}")
    logger.info(f"Case directory exists: {os.path.exists(case_dir)}")
    
    main_py_path = os.path.join(case_dir, "main.py")
    logger.info(f"Main.py path: {main_py_path}")
    logger.info(f"Main.py exists: {os.path.exists(main_py_path)}")
    
    start_cmd = "cd " + case_dir + ";python3 main.py "
    start_cmd += " --vendor=" + config.vendor
    start_cmd += " --case_name=" + op
    start_cmd += " --spectflops=" + spectflops
    start_cmd += " --dataformat=" + dataformat
    start_cmd += " --oplib=" + oplib
    start_cmd += " --chip=" + chip

    script_log_file = os.path.join(os.path.dirname(logfile),
                                   "operation.log.txt")

    logger.info(f"Command to execute: {start_cmd}")
    logger.info(f"Output will be written to: {script_log_file}")

    logger.info("Starting benchmark execution...")
    try:
        f = open(script_log_file, "w")
        logger.info(f"Opened output file: {script_log_file}")
        
        p = subprocess.Popen(start_cmd,
                             shell=True,
                             stdout=f,
                             stderr=subprocess.STDOUT)
        logger.info(f"Started subprocess with PID: {p.pid}")
        
        return_code = p.wait()
        f.close()
        
        logger.info(f"Subprocess finished with return code: {return_code}")
        
        # 检查输出文件是否有内容
        if os.path.exists(script_log_file):
            file_size = os.path.getsize(script_log_file)
            logger.info(f"Output file size: {file_size} bytes")
            if file_size > 0:
                with open(script_log_file, 'r') as rf:
                    content = rf.read()
                    logger.info(f"Output file content (first 500 chars): {content[:500]}")
            else:
                logger.warning("Output file is empty")
        else:
            logger.error("Output file was not created")
            
        if return_code == 0:
            logger.info("Task completed successfully")
        else:
            logger.error(f"Task failed with return code: {return_code}")
            
    except Exception as e:
        logger.error(f"Exception during benchmark execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("Container main finished")
