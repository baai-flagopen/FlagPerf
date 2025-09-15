#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
调试torch问题 - 检查为什么基于pytorch镜像的容器中没有torch
"""
import subprocess
import sys

def test_base_pytorch_image():
    """测试基础pytorch镜像"""
    print("=== 测试基础PyTorch镜像 ===")
    
    base_image = "cr.metax-tech.com/public-library/maca-c500-pytorch:2.33.0.6-torch2.4-py310-ubuntu22.04-amd64"
    
    print(f"1. 测试基础镜像: {base_image}")
    
    test_commands = [
        "python3 --version",
        "which python3",
        "python3 -c 'import sys; print(sys.path)'",
        "python3 -c 'import torch; print(f\"torch版本: {torch.__version__}\")'",
        "pip3 list | grep torch",
        "ls -la /usr/local/lib/python*/dist-packages/ | grep torch || echo 'No torch in dist-packages'",
        "find /usr -name '*torch*' 2>/dev/null | head -10 || echo 'torch files not found'"
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n1.{i} 执行: {cmd}")
        try:
            result = subprocess.run([
                "docker", "run", "--rm", 
                "--device=/dev/dri", "--device=/dev/mxcd", "--group-add", "video",
                base_image, "bash", "-c", cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✓ 成功: {result.stdout.strip()}")
            else:
                print(f"✗ 失败 (返回码: {result.returncode})")
                print(f"标准错误: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print("✗ 超时")
        except Exception as e:
            print(f"✗ 异常: {e}")

def test_flagperf_image():
    """测试flagperf镜像"""
    print("\n=== 测试FlagPerf镜像 ===")
    
    flagperf_image = "flagperf-operation-metax-ngctorch2403:t_v0.1"
    
    print(f"2. 测试FlagPerf镜像: {flagperf_image}")
    
    test_commands = [
        "python3 --version",
        "which python3",
        "python3 -c 'import sys; print(sys.path)'",
        "python3 -c 'import torch; print(f\"torch版本: {torch.__version__}\")'",
        "pip3 list | grep torch",
        "echo $PATH",
        "echo $PYTHONPATH",
        "ls -la /usr/local/lib/python*/dist-packages/ | grep torch || echo 'No torch in dist-packages'",
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n2.{i} 执行: {cmd}")
        try:
            result = subprocess.run([
                "docker", "run", "--rm", 
                "--device=/dev/dri", "--device=/dev/mxcd", "--group-add", "video",
                flagperf_image, "bash", "-c", cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✓ 成功: {result.stdout.strip()}")
            else:
                print(f"✗ 失败 (返回码: {result.returncode})")
                print(f"标准错误: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print("✗ 超时")
        except Exception as e:
            print(f"✗ 异常: {e}")

def compare_images():
    """比较两个镜像的差异"""
    print("\n=== 比较镜像差异 ===")
    
    base_image = "cr.metax-tech.com/public-library/maca-c500-pytorch:2.33.0.6-torch2.4-py310-ubuntu22.04-amd64"
    flagperf_image = "flagperf-operation-metax-ngctorch2403:t_v0.1"
    
    print("3. 检查镜像历史和层级差异")
    
    try:
        # 检查flagperf镜像的构建历史
        result = subprocess.run([
            "docker", "history", "--no-trunc", flagperf_image
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("FlagPerf镜像构建历史:")
            print(result.stdout)
        else:
            print(f"获取镜像历史失败: {result.stderr}")
            
    except Exception as e:
        print(f"比较镜像异常: {e}")

def check_dockerfile():
    """检查Dockerfile内容"""
    print("\n=== 检查Dockerfile ===")
    
    dockerfile_path = "/home/baai/FlagPerf/operation/vendors/metax/ngctorch2403/Dockerfile"
    
    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            print("当前Dockerfile内容:")
            print(content)
    except Exception as e:
        print(f"读取Dockerfile失败: {e}")

if __name__ == "__main__":
    print("开始调试torch问题...")
    
    test_base_pytorch_image()
    test_flagperf_image()
    compare_images()
    check_dockerfile()
    
    print("\n=== 调试完成 ===")
    print("\n可能的问题:")
    print("1. 基础镜像中的torch安装在特殊路径")
    print("2. 需要激活conda环境或设置特殊的PYTHONPATH")
    print("3. FlagPerf镜像构建过程中覆盖了torch安装")
    print("4. 需要特殊的环境变量或初始化脚本")
