#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
合并多算子执行后的正确性测试结果
在所有算子测试完成后调用，将各算子子目录的正确性日志合并到统一的result.json中
"""

import os
import sys
from argparse import ArgumentParser

# 添加path以便导入parse_log模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_log import finalize_correctness_logs_only


def parse_args():
    parser = ArgumentParser(description="Merge correctness results from multiple operation subdirectories")
    
    parser.add_argument("--timestamp_dir",
                        type=str,
                        required=True,
                        help="Timestamp directory containing operation subdirectories")
    
    return parser.parse_args()


def main():
    print("=== Starting Results Merge Tool ===")
    
    args = parse_args()
    timestamp_dir = args.timestamp_dir
    
    print(f"Target timestamp directory: {timestamp_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(timestamp_dir):
        print(f"Error: Timestamp directory does not exist: {timestamp_dir}")
        sys.exit(1)
    
    if not os.path.isdir(timestamp_dir):
        print(f"Error: Path is not a directory: {timestamp_dir}")
        sys.exit(1)
    
    # 执行合并
    try:
        summary = finalize_correctness_logs_only(timestamp_dir)
        
        if summary:
            print("\n=== Merge Summary ===")
            print(f"Timestamp Directory: {summary['timestamp_dir']}")
            print(f"Processed Cases: {len(summary['processed_cases'])}")
            for case in summary['processed_cases']:
                print(f"  - {case}")
            print(f"Total Correctness Entries: {summary['total_correctness_entries']}")
            print(f"Merged Log Path: {summary['merged_log_path']}")
            print("=== Correctness logs merge completed successfully ===")
            print("Note: result.json contains only performance data, correctness data is handled by MD formatting scripts")
        else:
            print("Error: Merge failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
