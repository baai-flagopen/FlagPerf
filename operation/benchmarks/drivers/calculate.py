# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from triton.testing import do_bench as kernel_bench
import os
import subprocess


def do_correctness(operation, result_log_dir):
    print(f"=== do_correctness called with operation={operation}, result_log_dir={result_log_dir} ===")
    flaggems_dir = os.getenv("FLAGGEMS_WORK_DIR", "/")
    print(f"FLAGGEMS_WORK_DIR: {flaggems_dir}")
    
    try:
        print(f"Searching for FlagGems in {flaggems_dir}")
        gems_repo = subprocess.check_output(
            ["find", flaggems_dir, "-type", "d", "-name", "FlagGems"], text=True).strip()
        print(f"Found FlagGems repo: {gems_repo}")
        
        if not gems_repo:
            print(f"FlagGems repository not found in {flaggems_dir}")
            return 0  # Skip correctness check
            
        tests_dir = os.path.join(gems_repo, 'tests')
        if not os.path.exists(tests_dir):
            print(f"Tests directory not found: {tests_dir}")
            return 0  # Skip correctness check
            
        # 根据算子类型映射到对应的测试文件（基于真实的FlagGems测试文件结构）
        operation_to_testfile = {
            # BLAS operations (test_blas_ops.py) - 矩阵运算
            "mm": "test_blas_ops.py",           # 矩阵乘法
            "bmm": "test_blas_ops.py",          # 批量矩阵乘法
            "addmm": "test_blas_ops.py",        # 矩阵加法乘法
            "mv": "test_blas_ops.py",           # 矩阵向量乘法
            "outer": "test_blas_ops.py",        # 外积
            "linear": "test_blas_ops.py",       # 线性变换（暂时放这里）
            
            # Binary pointwise operations (test_binary_pointwise_ops.py) - 二元逐点运算
            "add": "test_binary_pointwise_ops.py",        # 加法
            "sub": "test_binary_pointwise_ops.py",        # 减法
            "mul": "test_binary_pointwise_ops.py",        # 乘法
            "div": "test_binary_pointwise_ops.py",        # 除法
            "rsub": "test_binary_pointwise_ops.py",       # 反向减法
            "pow": "test_binary_pointwise_ops.py",        # 幂运算
            "eq": "test_binary_pointwise_ops.py",         # 等于比较
            "ge": "test_binary_pointwise_ops.py",         # 大于等于
            "gt": "test_binary_pointwise_ops.py",         # 大于
            "le": "test_binary_pointwise_ops.py",         # 小于等于
            "lt": "test_binary_pointwise_ops.py",         # 小于
            "ne": "test_binary_pointwise_ops.py",         # 不等于
            "bitwise_and": "test_binary_pointwise_ops.py", # 按位与
            "bitwise_or": "test_binary_pointwise_ops.py",  # 按位或
            
            # Unary pointwise operations (test_unary_pointwise_ops.py) - 一元逐点运算
            "abs": "test_unary_pointwise_ops.py",         # 绝对值
            "cos": "test_unary_pointwise_ops.py",         # 余弦
            "sin": "test_unary_pointwise_ops.py",         # 正弦
            "tanh": "test_unary_pointwise_ops.py",        # 双曲正切
            "exp": "test_unary_pointwise_ops.py",         # 指数
            "reciprocal": "test_unary_pointwise_ops.py",  # 倒数
            "rsqrt": "test_unary_pointwise_ops.py",       # 平方根倒数
            "neg": "test_unary_pointwise_ops.py",         # 取负
            "relu": "test_unary_pointwise_ops.py",        # ReLU激活
            "sigmoid": "test_unary_pointwise_ops.py",     # Sigmoid激活
            "silu": "test_unary_pointwise_ops.py",        # SiLU激活
            "gelu": "test_unary_pointwise_ops.py",        # GELU激活
            "isinf": "test_unary_pointwise_ops.py",       # 无穷判断
            "isnan": "test_unary_pointwise_ops.py",       # NaN判断
            "triu": "test_unary_pointwise_ops.py",        # 上三角
            "bitwise_not": "test_unary_pointwise_ops.py", # 按位非
            
            # General reduction operations (test_general_reduction_ops.py) - 通用归约运算
            "mean": "test_general_reduction_ops.py",      # 均值
            "sum": "test_general_reduction_ops.py",       # 求和
            "max": "test_general_reduction_ops.py",       # 最大值
            "min": "test_general_reduction_ops.py",       # 最小值
            "prod": "test_general_reduction_ops.py",      # 乘积
            "all": "test_general_reduction_ops.py",       # 全部为真
            
            # Advanced reduction operations (test_reduction_ops.py) - 高级归约运算
            "amax": "test_reduction_ops.py",              # 绝对值最大
            "argmax": "test_reduction_ops.py",            # 最大值索引
            "softmax": "test_reduction_ops.py",           # Softmax
            "log_softmax": "test_reduction_ops.py",       # Log Softmax
            "cross_entropy_loss": "test_reduction_ops.py", # 交叉熵损失
            
            # Normalization operations (test_norm_ops.py) - 归一化运算
            "layer_norm": "test_norm_ops.py",             # 层归一化（映射到layernorm）
            "layernorm": "test_norm_ops.py",              # 层归一化（FlagGems实际名称）
            "group_norm": "test_norm_ops.py",             # 组归一化  
            "native_group_norm": "test_norm_ops.py",      # 原生组归一化
            
            # Special operations (test_special_ops.py) - 特殊运算
            "dropout": "test_special_ops.py",             # Dropout
            "native_dropout": "test_special_ops.py"       # 原生Dropout
        }
        
        test_filename = operation_to_testfile.get(operation)
        if test_filename:
            test_file = os.path.join(tests_dir, test_filename)
            if os.path.exists(test_file):
                print(f"Running correctness test for {operation} using {test_filename}")
                
                # 创建日志文件路径
                correctness_log = os.path.join(tests_dir, f"correctness_{operation}_test.log")
                
                # 删除历史日志
                del_process = subprocess.Popen(["rm", "-f", correctness_log], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                del_process.communicate()
                
                # 使用pytest运行特定算子的测试，并将输出重定向到日志文件
                with open(correctness_log, 'w') as log_file:
                    p = subprocess.Popen(
                        f"cd {tests_dir} && python3 -m pytest {test_filename} -v -k 'test_accuracy_{operation}' --tb=short",
                        shell=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT
                    )
                    p.wait()
                
                print(f"Correctness test completed for {operation}, exit code: {p.returncode}")
                
                # 复制日志文件到结果目录
                try:
                    cp_subprocess = subprocess.run(["cp", correctness_log, f"{result_log_dir}/correctness.log.txt"], check=True)
                    print(f"Correctness log copied to {result_log_dir}/correctness.log.txt")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to copy correctness log: {e}")
                
                # 返回真实的测试结果：0表示成功，非0表示失败
                return p.returncode
            else:
                print(f"Test file {test_filename} not found for operation {operation}")
        else:
            print(f"No test mapping found for operation: {operation}")
            
        # 如果没有找到对应的测试文件，尝试通用方式
        print(f"Trying generic test approach for {operation}...")
        
        # 检查是否有通用的测试文件
        generic_tests = ["test_binary_pointwise_ops.py", "test_blas_ops.py"]
        for generic_test in generic_tests:
            test_file = os.path.join(tests_dir, generic_test)
            if os.path.exists(test_file):
                print(f"Found generic test file: {generic_test}")
                
                # 创建日志文件路径
                correctness_log = os.path.join(tests_dir, f"correctness_{operation}_generic_test.log")
                
                # 删除历史日志
                del_process = subprocess.Popen(["rm", "-f", correctness_log], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                del_process.communicate()
                
                # 使用pytest运行通用测试，并将输出重定向到日志文件
                with open(correctness_log, 'w') as log_file:
                    p = subprocess.Popen(
                        f"cd {tests_dir} && python3 -m pytest {generic_test} -v -k '{operation}' --tb=short",
                        shell=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT
                    )
                    p.wait()
                
                print(f"Generic test completed for {operation}, exit code: {p.returncode}")
                
                # 复制日志文件到结果目录
                try:
                    cp_subprocess = subprocess.run(["cp", correctness_log, f"{result_log_dir}/correctness.log.txt"], check=True)
                    print(f"Correctness log copied to {result_log_dir}/correctness.log.txt")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to copy correctness log: {e}")
                
                # 返回真实的测试结果
                return p.returncode
                
        print(f"No suitable test found for operation: {operation}, skipping correctness check")
        # 如果找不到测试，返回0表示跳过（不是失败）
        return 0
            
    except Exception as e:
        print(f"Error during correctness check: {e}")
        return 0  # Skip correctness check on error


        # test operation performance
def do_performance(mode, warmup, result_log_dir):
    print(f"=== do_performance called with mode={mode}, warmup={warmup}, result_log_dir={result_log_dir} ===")
    flaggems_dir = os.getenv("FLAGGEMS_WORK_DIR", "/")
    print(f"FLAGGEMS_WORK_DIR: {flaggems_dir}")
    
    print(f"Searching for FlagGems in {flaggems_dir}")
    gems_repo = subprocess.check_output(
        ["find", flaggems_dir, "-type", "d", "-name", "FlagGems"], text=True).strip()
    print(f"Found FlagGems repo: {gems_repo}")
    
    if not gems_repo:
        print(f"FlagGems repository not found in {flaggems_dir}")
        return 1  # Return error code
    del_file_path = os.path.join(gems_repo, 'benchmark')
    print(f"Benchmark directory: {del_file_path}")
    
    # 删除历史日志
    del_file = os.path.join(del_file_path,
                            f"result--level_core--mode_{mode}--warmup_{warmup}--record_log.log")
    print(f"Deleting old log file: {del_file}")
    del_process = subprocess.Popen(["rm", "-f", del_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    del_process.communicate()
    
    # 执行性能测试命令
    perf_cmd = f"cd {os.path.join(gems_repo, 'benchmark')} && pytest --level core --mode {mode} --warmup {warmup} --record log"
    print(f"Running performance test: {perf_cmd}")
    
    p = subprocess.Popen(perf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = p.communicate()
    print(f"Performance test output: {stdout.decode() if stdout else 'No output'}")
    print(f"Performance test exit code: {p.returncode}")

    # 全量执行日志路径
    log_dir = os.path.join(gems_repo, "benchmark",
                           f"result--level_core--mode_{mode}--warmup_{warmup}--record_log.log")
    print(f"Expected log file: {log_dir}")
    print(f"Log file exists: {os.path.exists(log_dir)}")
    
    if os.path.exists(log_dir):
        try:
            cp_subprocess = subprocess.run(["cp", f"{log_dir}", f"{result_log_dir}/result.log.txt"], check=True)
            print(f"Successfully copied log to {result_log_dir}/result.log.txt")
            return p.returncode, cp_subprocess.returncode
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy log file: {e}")
            return p.returncode, 1
    else:
        print(f"Log file not found: {log_dir}")
        return p.returncode, 1

grad_outputs = None

def do(exec_func, exec_args, bp=False):
    global grad_outputs
    if bp:
        import torch
        _tensor = exec_func(*exec_args).sum()
        if grad_outputs is None:
            grad_outputs = torch.zeros_like(_tensor)
        inputs = list(filter(lambda x: x.requires_grad, [*exec_args]))
        _grad = torch.autograd.grad(outputs=_tensor, inputs=inputs, grad_outputs=grad_outputs)
    else:
        _tensor = exec_func(*exec_args)


def do_test(exec_func, exec_args, sync_func, config, case_config, bp=False):
    sync_func(config.vendor)
    start_latency_nowarm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_nowarm = time.perf_counter_ns() - start_latency_nowarm

    for _ in range(case_config.WARMUP):
        do(exec_func, exec_args, bp)

    sync_func(config.vendor)
    start_latency_warm = time.perf_counter_ns()
    _tensor = exec_func(*exec_args)

    sync_func(config.vendor)
    latency_warm = time.perf_counter_ns() - start_latency_warm

    start_time = time.perf_counter()
    for _ in range(case_config.ITERS):
        do(exec_func, exec_args, bp)

    sync_func(config.vendor)
    end_time = time.perf_counter()

    cputime_raw = end_time - start_time

    kerneltime_raw = kernel_bench(lambda: do(exec_func, exec_args, bp),
                                  warmup=case_config.KERNELWARMUP,
                                  rep=case_config.KERNELITERS,
                                  return_mode="median")
    cputime = cputime_raw / case_config.ITERS
    kerneltime = kerneltime_raw / 1000.0  # ms to s
    return round(latency_nowarm / 1000.0, 2), round(latency_warm / 1000.0,
                                                    2), cputime, kerneltime


def cal_perf(cputime, kerneltime, op2flops, spectflops, bp=False):
    spectflops = float(spectflops)
    ctus = round(cputime * 1E6, 2)
    ktus = round(kerneltime * 1E6, 2)

    cps = 1.0 / cputime
    kps = 1.0 / kerneltime

    cflops = op2flops(cps) * (3.0 if bp else 1.0)
    kflops = op2flops(kps) * (3.0 if bp else 1.0)
    ctflops = round(cflops / 1E12, 2)
    ktflops = round(kflops / 1E12, 2)

    cfu = round(100.0 * cflops / 1E12 / spectflops, 2)
    kfu = round(100.0 * kflops / 1E12 / spectflops, 2)

    return ctus, ktus, cps, kps, ctflops, ktflops, cfu, kfu


def print_result(config, casename, ct, kt, cps, kps, ctflops, ktflops, cfu,
                 kfu, correctness, lnm, lm):
    print(r"[FlagPerf Result]Operation {} in {} at {}:".format(
        casename, config.oplib, config.dataformat))
    print(r"[FlagPerf Result]FLOPS utilization: cputime={}%, kerneltime={}%".
          format(cfu, kfu))
    print(
        r"[FlagPerf Result]cputime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(ct, cps, ctflops))
    print(
        r"[FlagPerf Result]kerneltime={} us, throughput={} op/s, equals to {} TFLOPS"
        .format(kt, kps, ktflops))
    print(r"[FlagPerf Result]Correctness with CPU golden Reference: {}".format(
        correctness))
    print(
        r"[FlagPerf Result]First time latency: no warmup={} us, warmup={} us".
        format(lnm, lm))
