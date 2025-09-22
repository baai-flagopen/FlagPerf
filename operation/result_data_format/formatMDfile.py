import os
import re


def render(extracted_values, readme_file_path, vendor, shm_size, chip, all_correctness_logs=None):
    json_data = []
    for key, value in extracted_values.items():
        json_data.append(value)
    dest_file_path = os.path.join(readme_file_path, "README.md")

    # 合并所有正确性日志的结果
    correctness_result = merge_all_correctness_logs(all_correctness_logs) if all_correctness_logs else None

    # 生成Markdown内容
    markdown_content = create_markdown_content(json_data, vendor, shm_size, chip, correctness_result)
    with open(dest_file_path, 'w') as file:
        file.write(markdown_content)


def create_markdown_content(data, vendor, shm_size, chip, correctness_result):
    """
    创建完整的Markdown内容，包括性能测试表格和单独的正确性测试章节
    """
    v_chip = f'{vendor}_{chip}'
    content = f"# 参评AI芯片信息\n\n * 厂商：{vendor}\n * 产品名称：{v_chip}\n * 产品型号：{chip}\n * SHM_SIZE：{shm_size}\n\n"

    # 添加正确性测试结果章节
    content += "# 正确性测试结果\n\n"
    if correctness_result:
        content += f"**测试状态**: {correctness_result['status']}\n\n"
        content += f"**总体结果**: {correctness_result['summary']}\n\n"

        # 如果有按算子分组的结果，显示详细信息
        if correctness_result.get('algorithm_results'):
            content += "## 各算子测试结果\n\n"
            for algo_name, algo_result in correctness_result['algorithm_results'].items():
                algo_status_emoji = "✅" if "通过" in algo_result['status'] else "❌" if "失败" in algo_result['status'] else "⚠️"
                content += f"### {algo_status_emoji} {algo_name}\n\n"
                content += f"**状态**: {algo_result['status']}\n\n"
                content += f"**结果**: {algo_result['summary']}\n\n"
                
                if algo_result.get('test_results'):
                    content += "**详细测试项**:\n\n"
                    for test_name, test_result in algo_result['test_results'].items():
                        status_emoji = "✅" if test_result == "PASSED" else "❌" if test_result == "FAILED" else "⚠️"
                        content += f"- {status_emoji} {test_name}: {test_result}\n"
                    content += "\n"
        elif correctness_result.get('test_results'):
            # 如果没有按算子分组，显示所有测试结果
            content += "**各测试项结果**:\n\n"
            for test_name, result in correctness_result['test_results'].items():
                status_emoji = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
                content += f"- {status_emoji} {test_name}: {result}\n"
            content += "\n"
    else:
        content += "**测试状态**: ➖ 未进行正确性测试\n\n"
        content += "**说明**: 未找到正确性测试日志文件\n\n"

    # 添加性能测试结果表格
    content += "# 评测结果\n\n"
    content += "| op_name | dtype | shape_detail | 无预热时延(Latency-No warmup) | 预热时延(Latency-Warmup) | 原始吞吐(Raw-Throughput)| 核心吞吐(Core-Throughput) | 实际算力开销 | 实际算力利用率 | 实际算力开销(内核时间) | 实际算力利用率(内核时间) |\n"
    content += "| --- | ---| --- | ---| --- | ---| --- | ---| --- | ---| --- |\n"

    for row in data:
        content += f"| {row.get('op_name', 'N/A')} | {row.get('dtype', 'N/A')} | {row.get('shape_detail', 'N/A')} | {row.get('no_warmup_latency', 'N/A')} | {row.get('warmup_latency', 'N/A')} | {row.get('raw_throughput', 'N/A')} | {row.get('core_throughput', 'N/A')} | {row.get('ctflops', 'N/A')} | {row.get('cfu', 'N/A')} | {row.get('ktflops', 'N/A')} | {row.get('kfu', 'N/A')} |\n"

    return content


def merge_all_correctness_logs(correctness_log_paths):
    """
    合并多个correctness.log.txt文件的结果
    """
    if not correctness_log_paths:
        return None
    
    merged_result = {
        'status': '✅ 测试通过',
        'summary': '',
        'details': '',
        'test_results': {},
        'total_passed': 0,
        'total_failed': 0,
        'total_errors': 0,
        'algorithm_results': {}  # 按算子分组的结果
    }
    
    for log_path in correctness_log_paths:
        print(f"Processing correctness log: {log_path}")
        
        # 从路径中提取算子名称
        # 路径格式: .../timestamp/opv2:mm:FP16:312:flaggems:A100_40_SXM/hostname_noderank0/correctness.log.txt
        path_parts = log_path.split(os.sep)
        algorithm_name = "unknown"
        for part in reversed(path_parts):
            if ":" in part and not "noderank" in part:  # 算子名称格式包含冒号
                algorithm_name = part
                break
        
        # 解析单个日志文件
        single_result = parse_single_correctness_log(log_path)
        if single_result:
            # 统计总数
            merged_result['total_passed'] += single_result.get('passed_count', 0)
            merged_result['total_failed'] += single_result.get('failed_count', 0)
            merged_result['total_errors'] += single_result.get('error_count', 0)
            
            # 合并测试结果，添加算子前缀避免冲突
            for test_name, test_status in single_result.get('test_results', {}).items():
                merged_test_name = f"{algorithm_name}::{test_name}"
                merged_result['test_results'][merged_test_name] = test_status
            
            # 保存按算子分组的结果
            merged_result['algorithm_results'][algorithm_name] = {
                'status': single_result.get('status', '❓ 未知'),
                'summary': single_result.get('summary', ''),
                'passed_count': single_result.get('passed_count', 0),
                'failed_count': single_result.get('failed_count', 0),
                'error_count': single_result.get('error_count', 0),
                'test_results': single_result.get('test_results', {})
            }
            
            # 累积详细信息
            if merged_result['details']:
                merged_result['details'] += f"\n\n=== {algorithm_name} ===\n"
            else:
                merged_result['details'] = f"=== {algorithm_name} ===\n"
            merged_result['details'] += single_result.get('details', '')
    
    # 根据总体结果确定状态
    if merged_result['total_failed'] > 0 or merged_result['total_errors'] > 0:
        merged_result['status'] = '❌ 部分测试失败'
    elif merged_result['total_passed'] > 0:
        merged_result['status'] = '✅ 全部测试通过'
    else:
        merged_result['status'] = '⚠️ 无测试结果'
    
    # 生成总结
    merged_result['summary'] = f"总计: {merged_result['total_passed']} 个通过, {merged_result['total_failed']} 个失败, {merged_result['total_errors']} 个错误"
    
    return merged_result


def parse_single_correctness_log(log_path):
    """
    解析单个correctness.log.txt文件，提取pytest测试结果
    """
    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # 解析pytest输出
        result = {
            'status': '❓ 未知',
            'summary': '',
            'details': log_content.strip(),
            'test_results': {},
            'passed_count': 0,
            'failed_count': 0,
            'error_count': 0
        }

        lines = log_content.split('\n')

        # 查找测试结果摘要行和具体测试结果
        summary_pattern = r'=+ (.+) =+'
        test_pattern = r'(.+::test_accuracy_\w+\[.*?\])\s+(PASSED|FAILED|ERROR)\s+\[\s*\d+%\]'

        for line in lines:
            line = line.strip()

            # 提取各个测试项的结果
            test_match = re.search(test_pattern, line)
            if test_match:
                test_name = test_match.group(1).split("::")[-1]  # 只取测试函数名部分
                test_result = test_match.group(2)
                result['test_results'][test_name] = test_result
                
                if test_result == "PASSED":
                    result['passed_count'] += 1
                elif test_result == "FAILED":
                    result['failed_count'] += 1
                elif test_result == "ERROR":
                    result['error_count'] += 1
            
            # 查找总结行
            summary_match = re.search(summary_pattern, line)
            if summary_match and ('passed' in line or 'failed' in line):
                summary_text = summary_match.group(1)
                result['summary'] = summary_text
                break
        
        # 确定状态
        if result['failed_count'] > 0 or result['error_count'] > 0:
            result['status'] = '❌ 测试失败'
        elif result['passed_count'] > 0:
            result['status'] = '✅ 测试通过'
        else:
            result['status'] = '⚠️ 无测试结果'
        
        # 如果没有找到总结，根据计数生成
        if not result['summary']:
            result['summary'] = f"{result['passed_count']} 个通过, {result['failed_count']} 个失败, {result['error_count']} 个错误"
        
        return result
        
    except Exception as e:
        return {
            'status': '⚠️ 解析错误',
            'summary': f'解析correctness.log.txt时发生错误: {str(e)}',
            'details': '',
            'test_results': {},
            'passed_count': 0,
            'failed_count': 0,
            'error_count': 0
        }


def parse_correctness_log(result_path):
    """
    解析correctness.log.txt文件，提取pytest测试结果
    """
    correctness_log_path = os.path.join(result_path, "correctness.log.txt")

    if not os.path.exists(correctness_log_path):
        return None

    try:
        with open(correctness_log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # 解析pytest输出
        result = {
            'status': '❓ 未知',
            'summary': '',
            'details': log_content.strip(),
            'test_results': {}
        }

        lines = log_content.split('\n')

        # 查找测试结果摘要行，如："========== 9 passed, 423 deselected, 12 warnings in 174.55s (0:02:54) =========="
        summary_pattern = r'=+ (.+) =+'
        test_pattern = r'(.+::test_accuracy_\w+\[.*?\])\s+(PASSED|FAILED|ERROR)\s+\[\s*\d+%\]'

        passed_count = 0
        failed_count = 0
        error_count = 0

        for line in lines:
            line = line.strip()

            # 提取各个测试项的结果
            test_match = re.search(test_pattern, line)
            if test_match:
                test_name = test_match.group(1).split("::")[-1]  # 只取测试函数名部分
                test_result = test_match.group(2)
                result['test_results'][test_name] = test_result
                
                if test_result == "PASSED":
                    passed_count += 1
                elif test_result == "FAILED":
                    failed_count += 1
                elif test_result == "ERROR":
                    error_count += 1
            
            # 查找总结行
            summary_match = re.search(summary_pattern, line)
            if summary_match and ('passed' in line or 'failed' in line):
                summary_text = summary_match.group(1)
                result['summary'] = summary_text
                
                # 根据总结确定整体状态
                if failed_count > 0 or error_count > 0:
                    result['status'] = '❌ 测试失败'
                elif passed_count > 0:
                    result['status'] = '✅ 测试通过'
                else:
                    result['status'] = '⚠️ 无测试结果'
                break
        
        # 如果没有找到总结行，根据计数判断
        if not result['summary']:
            if passed_count > 0 and failed_count == 0 and error_count == 0:
                result['status'] = '✅ 测试通过'
                result['summary'] = f"{passed_count} 个测试通过"
            elif failed_count > 0 or error_count > 0:
                result['status'] = '❌ 测试失败'
                result['summary'] = f"{passed_count} 个通过, {failed_count} 个失败, {error_count} 个错误"
            else:
                result['status'] = '⚠️ 无测试结果'
                result['summary'] = "未找到有效的测试结果"
        
        return result
        
    except Exception as e:
        return {
            'status': '⚠️ 解析错误',
            'summary': f'解析correctness.log.txt时发生错误: {str(e)}',
            'details': '',
            'test_results': {}
        }