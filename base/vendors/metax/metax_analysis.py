def analysis_log(logpath, config):
    logfile = open(logpath)

    result = {"temp": {}, "power": {}, "mem": {}}
    for gpuID in range(config.NPROC_PER_NODE):
        for monitor_index in result.keys():
            result[monitor_index][gpuID] = []

    max_mem = None
    next_gpu_id = 0
    for line in logfile.readlines():
        # 适配新格式: 35C 52W | 858/65536
        if "C " in line and "W " in line and "|" in line and "/" in line:
            # 跳过空行和时间戳行
            line = line.strip()
            if not line or line.startswith("2025-") or line == "|":
                continue
                
            try:
                # 分割格式: "35C 52W | 858/65536"
                parts = line.split(" | ")
                if len(parts) != 2:
                    continue
                    
                temp_power_part = parts[0].strip()  # "35C 52W"
                mem_part = parts[1].strip()         # "858/65536"
                
                # 解析温度和功率
                temp_power_items = temp_power_part.split()
                if len(temp_power_items) != 2:
                    continue
                    
                temp_str = temp_power_items[0]  # "35C"
                power_str = temp_power_items[1] # "52W"
                
                # 提取数值
                temp = float(temp_str[:-1])     # 去掉 "C"
                power = float(power_str[:-1])   # 去掉 "W"
                
                # 解析内存使用情况
                if "/" in mem_part:
                    usage_and_maxusage = mem_part
                    usage = float(usage_and_maxusage.split("/")[0])
                    max_mem_current = float(usage_and_maxusage.split("/")[1])
                    
                    if max_mem is None:
                        max_mem = max_mem_current
                        result["max_mem"] = max_mem
                    
                    # 添加数据到结果中
                    result["temp"][next_gpu_id].append(temp)
                    result["power"][next_gpu_id].append(power)
                    result["mem"][next_gpu_id].append(usage)
                    next_gpu_id = (next_gpu_id + 1) % config.NPROC_PER_NODE
                    
            except (ValueError, IndexError) as e:
                # 解析失败时跳过该行
                continue

    return result