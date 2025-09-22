import os
import re


def render(extracted_values, readme_file_path, vendor, shm_size, chip):
    json_data = []
    for key, value in extracted_values.items():
        json_data.append(value)
    dest_file_path = os.path.join(readme_file_path, "README.md")

    # 直接使用parse_correctness_log从文件解析正确性结果
    correctness_result = parse_correctness_log(readme_file_path)

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

        if correctness_result.get('test_results'):
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