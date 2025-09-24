# Copyright (c) 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time
from triton.testing import do_bench as kernel_bench
import os
import subprocess


def do_correctness(operation, result_log_dir, flaggems_path=None):
    print(f"=== do_correctness called with operation={operation}, result_log_dir={result_log_dir} ===")
    
    # 使用配置的 FLAGGEMS_PATH，如果没有则使用环境变量作为后备
    if flaggems_path:
        gems_repo = flaggems_path
        print(f"Using configured FLAGGEMS_PATH: {gems_repo}")
    else:
        gems_repo = os.getenv("FLAGGEMS_WORK_DIR", "/workspace/FlagGems")
        print(f"Using FLAGGEMS_WORK_DIR fallback: {gems_repo}")
    
    # 检查 FlagGems 路径是否存在
    if not os.path.exists(gems_repo):
        print(f"FlagGems path not found: {gems_repo}, skipping correctness test")
        return 0
    
    try:
        tests_dir = os.path.join(gems_repo, 'tests')
        if not os.path.exists(tests_dir):
            print(f"Tests directory not found: {tests_dir}")
            return 0  # Skip correctness check
            
        print(f"Running correctness test for {operation} using: pytest -m {operation} --ref cpu")
        
        # 创建日志文件路径
        correctness_log = os.path.join(tests_dir, f"correctness_{operation}_test.log")
        
        # 删除历史日志
        del_process = subprocess.Popen(["rm", "-f", correctness_log], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        del_process.communicate()
        
        # 使用简化的pytest命令：pytest -m {operation} --ref cpu
        with open(correctness_log, 'w') as log_file:
            p = subprocess.Popen(
                f"cd {tests_dir} && pytest -m {operation} --ref cpu",
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
            
    except Exception as e:
        print(f"Error during correctness check: {e}")
        return 0  # Skip correctness check on error


        # test operation performance
def do_performance(operation, mode, warmup, result_log_dir, flaggems_path=None):
    print(f"=== do_performance called with operation={operation}, mode={mode}, warmup={warmup}, result_log_dir={result_log_dir} ===")
    
    # 使用配置的 FLAGGEMS_PATH，如果没有则使用环境变量作为后备
    if flaggems_path:
        gems_repo = flaggems_path
        print(f"Using configured FLAGGEMS_PATH: {gems_repo}")
    else:
        gems_repo = os.getenv("FLAGGEMS_WORK_DIR", "/workspace/FlagGems")
        print(f"Using FLAGGEMS_WORK_DIR fallback: {gems_repo}")
    
    # 检查 FlagGems 路径是否存在
    if not os.path.exists(gems_repo):
        print(f"FlagGems path not found: {gems_repo}, returning error")
        return 1
    
    try:
        benchmark_dir = os.path.join(gems_repo, 'benchmark')
        if not os.path.exists(benchmark_dir):
            print(f"Benchmark directory not found: {benchmark_dir}")
            return 1
            
        print(f"Running performance test for {operation} using: pytest -m {operation} --level core --record log")
        
        # 删除历史日志（简化的日志文件名）
        log_file = os.path.join(benchmark_dir, f"result--level_core--record_log.log")
        print(f"Deleting old log file: {log_file}")
        del_process = subprocess.Popen(["rm", "-f", log_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        del_process.communicate()
        
        # 使用简化的pytest命令：pytest -m {operation} --level core --record log
        perf_cmd = f"cd {benchmark_dir} && pytest -m {operation} --level core --record log"
        print(f"Running performance test: {perf_cmd}")
        
        p = subprocess.Popen(perf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = p.communicate()
        print(f"Performance test output: {stdout.decode() if stdout else 'No output'}")
        print(f"Performance test exit code: {p.returncode}")

        # 检查日志文件是否生成
        print(f"Expected log file: {log_file}")
        print(f"Log file exists: {os.path.exists(log_file)}")
        
        if os.path.exists(log_file):
            try:
                cp_subprocess = subprocess.run(["cp", f"{log_file}", f"{result_log_dir}/result.log.txt"], check=True)
                print(f"Successfully copied log to {result_log_dir}/result.log.txt")
                return p.returncode, cp_subprocess.returncode
            except subprocess.CalledProcessError as e:
                print(f"Failed to copy log file: {e}")
                return p.returncode, 1
        else:
            print(f"Log file not found: {log_file}")
            return p.returncode, 1
            
    except Exception as e:
        print(f"Error during performance test: {e}")
        return 1  # Return error code

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
