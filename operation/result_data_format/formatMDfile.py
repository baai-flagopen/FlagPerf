import os


def render(extracted_values, readme_file_path, vendor, shm_size, chip):
    json_data = []
    for key, value in extracted_values.items():
        json_data.append(value)
    dest_file_path = os.path.join(readme_file_path, "README.md")
    markdown_table = creat_markdown_table(json_data, vendor, shm_size, chip)
    with open(dest_file_path, 'w') as file:
        file.write(markdown_table)


def creat_markdown_table(data, vendor, shm_size, chip):
    v_chip = f'{vendor}_{chip}'
    table = f"# 参评AI芯片信息\n\n * 厂商：{vendor}\n * 产品名称：{v_chip}\n * 产品型号：{chip}\n * SHM_SIZE：{shm_size}\n\n\n\n"
    table += "# 评测结果\n\n"
    table += "| op_name | dtype | shape_detail | 无预热时延(Latency-No warmup) | 预热时延(Latency-Warmup) | 原始吞吐(Raw-Throughput)| 核心吞吐(Core-Throughput) | 实际算力开销 | 实际算力利用率 | 实际算力开销(内核时间) | 实际算力利用率(内核时间) | 正确性测试状态 | 正确性测试详情 |\n| --- | ---| --- | ---| --- | ---| --- | ---| --- | ---| --- | ---| --- |\n"
    for row in data:
        # 获取正确性测试结果，如果不存在则使用默认值
        correctness_status = row.get('correctness_status', 'not_tested')
        correctness_details = row.get('correctness_details', 'No correctness test data')
        
        # 格式化正确性状态显示
        status_display = format_correctness_status(correctness_status)
        
        table += f"| {row.get('op_name', 'N/A')} | {row.get('dtype', 'N/A')} | {row.get('shape_detail', 'N/A')} | {row.get('no_warmup_latency', 'N/A')} | {row.get('warmup_latency', 'N/A')} | {row.get('raw_throughput', 'N/A')} | {row.get('core_throughput', 'N/A')} | {row.get('ctflops', 'N/A')} | {row.get('cfu', 'N/A')} | {row.get('ktflops', 'N/A')} | {row.get('kfu', 'N/A')} | {status_display} | {correctness_details} |\n"
    return table


def format_correctness_status(status):
    """
    格式化正确性测试状态为更友好的显示
    """
    status_map = {
        'passed': '✅ 通过',
        'failed': '❌ 失败', 
        'error': '⚠️ 错误',
        'not_run': '⏸️ 未运行',
        'not_tested': '➖ 未测试',
        'not_found': '❓ 未找到',
        'unknown': '❔ 未知'
    }
    return status_map.get(status, f'❔ {status}')