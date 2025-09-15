#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
简化的容器测试脚本，用于诊断container_main.py执行问题
"""
import os
import sys
import subprocess
import time

def test_basic_execution():
    """测试基本的Python执行"""
    print("=== 测试基本Python执行 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")
    
def test_container_main_import():
    """测试container_main.py是否可以导入"""
    print("\n=== 测试container_main.py导入 ===")
    try:
        container_main_path = "/home/baai/FlagPerf/operation/container_main.py"
        print(f"container_main.py路径: {container_main_path}")
        print(f"文件存在: {os.path.exists(container_main_path)}")
        
        if os.path.exists(container_main_path):
            # 尝试执行语法检查
            result = subprocess.run([sys.executable, "-m", "py_compile", container_main_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ container_main.py语法检查通过")
            else:
                print(f"✗ container_main.py语法错误: {result.stderr}")
        
    except Exception as e:
        print(f"✗ 导入测试失败: {e}")

def test_loguru_import():
    """测试loguru导入"""
    print("\n=== 测试loguru导入 ===")
    try:
        import loguru
        print("✓ loguru导入成功")
        print(f"loguru版本: {loguru.__version__}")
    except ImportError as e:
        print(f"✗ loguru导入失败: {e}")

def test_directory_creation():
    """测试目录创建"""
    print("\n=== 测试目录创建 ===")
    test_dir = "/home/baai/FlagPerf/operation/result/test_run"
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ 目录创建成功: {test_dir}")
        print(f"目录存在: {os.path.exists(test_dir)}")
        
        # 测试写入文件
        test_file = os.path.join(test_dir, "test.log")
        with open(test_file, 'w') as f:
            f.write("Test log entry\n")
        print(f"✓ 测试文件创建成功: {test_file}")
        
    except Exception as e:
        print(f"✗ 目录创建失败: {e}")

def test_container_main_execution():
    """测试container_main.py直接执行"""
    print("\n=== 测试container_main.py直接执行 ===")
    
    # 构建最简单的参数
    cmd = [
        sys.executable, 
        "/home/baai/FlagPerf/operation/container_main.py",
        "--vendor", "metax",
        "--case_name", "mm:FP16:312:nativetorch:C550_64", 
        "--nnodes", "1",
        "--perf_path", "/home/baai/FlagPerf/operation",
        "--nproc_per_node", "8",
        "--log_dir", "/home/baai/FlagPerf/operation/result/test_run",
        "--log_level", "INFO",
        "--master_port", "29501",
        "--master_addr", "localhost",
        "--host_addr", "localhost", 
        "--node_rank", "0"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"返回码: {result.returncode}")
        print(f"标准输出: {result.stdout}")
        print(f"标准错误: {result.stderr}")
        
        if result.returncode == 0:
            print("✓ container_main.py执行成功")
        else:
            print(f"✗ container_main.py执行失败，返回码: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("✗ container_main.py执行超时")
    except Exception as e:
        print(f"✗ container_main.py执行异常: {e}")

if __name__ == "__main__":
    print("开始容器诊断测试...")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_basic_execution()
    test_loguru_import()
    test_directory_creation()
    test_container_main_import()
    test_container_main_execution()
    
    print("\n=== 诊断测试完成 ===")
