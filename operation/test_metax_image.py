#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
直接测试metax镜像的内容
"""
import subprocess
import sys

def test_metax_image_directly():
    """直接测试metax镜像中的torch环境"""
    print("=== 直接测试metax镜像 ===")
    
    image_name = "flagperf-operation-metax-ngctorch2403:t_v0.1"
    
    # 测试1: 检查镜像是否存在
    print(f"1. 检查镜像是否存在: {image_name}")
    try:
        result = subprocess.run([
            "docker", "images", "--format", "table {{.Repository}}:{{.Tag}}", 
            "--filter", f"reference={image_name}"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and image_name in result.stdout:
            print(f"✓ 镜像存在: {image_name}")
        else:
            print(f"✗ 镜像不存在: {image_name}")
            print(f"可用镜像:")
            list_result = subprocess.run(["docker", "images"], capture_output=True, text=True)
            print(list_result.stdout)
            return False
    except Exception as e:
        print(f"✗ 检查镜像失败: {e}")
        return False
    
    # 测试2: 在镜像中测试Python和torch
    print(f"\n2. 在镜像中测试Python环境")
    test_commands = [
        "python3 --version",
        "python3 -c 'import sys; print(f\"Python路径: {sys.executable}\")'",
        "python3 -c 'import torch; print(f\"torch版本: {torch.__version__}\")'",
        "python3 -c 'import torch; x=torch.randn(2,3); y=torch.randn(3,2); z=torch.mm(x,y); print(f\"torch.mm测试成功，结果形状: {z.shape}\")'",
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n2.{i} 执行: {cmd}")
        try:
            result = subprocess.run([
                "docker", "run", "--rm", 
                "--device=/dev/dri", "--device=/dev/mxcd", "--group-add", "video",
                image_name, "bash", "-c", cmd
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✓ 成功: {result.stdout.strip()}")
            else:
                print(f"✗ 失败 (返回码: {result.returncode})")
                print(f"标准输出: {result.stdout}")
                print(f"标准错误: {result.stderr}")
                
                if "torch" in cmd and "No module named" in result.stderr:
                    print("🔍 发现问题: 镜像中确实缺少torch模块")
                    return False
                    
        except subprocess.TimeoutExpired:
            print("✗ 超时")
        except Exception as e:
            print(f"✗ 异常: {e}")
    
    # 测试3: 测试flaggems
    print(f"\n3. 测试flaggems")
    flaggems_cmd = "python3 -c 'import flag_gems; print(f\"flaggems导入成功\"); flag_gems.enable(); print(f\"flaggems启用成功\")'"
    try:
        result = subprocess.run([
            "docker", "run", "--rm",
            "--device=/dev/dri", "--device=/dev/mxcd", "--group-add", "video", 
            image_name, "bash", "-c", flaggems_cmd
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✓ flaggems测试成功: {result.stdout.strip()}")
        else:
            print(f"⚠ flaggems测试失败: {result.stderr}")
    except Exception as e:
        print(f"⚠ flaggems测试异常: {e}")
    
    # 测试4: 测试benchmark执行
    print(f"\n4. 测试benchmark main.py语法")
    benchmark_cmd = "cd /home/baai/FlagPerf/operation/benchmarks/mm && python3 -m py_compile main.py"
    try:
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", "/home/baai/FlagPerf/operation:/home/baai/FlagPerf/operation",
            "--device=/dev/dri", "--device=/dev/mxcd", "--group-add", "video",
            image_name, "bash", "-c", benchmark_cmd
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✓ benchmark语法检查成功")
        else:
            print(f"✗ benchmark语法检查失败: {result.stderr}")
    except Exception as e:
        print(f"✗ benchmark测试异常: {e}")
    
    return True

if __name__ == "__main__":
    print("开始测试metax镜像...")
    test_metax_image_directly()
    print("\n=== 测试完成 ===")
