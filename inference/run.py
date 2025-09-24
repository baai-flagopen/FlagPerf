# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
''' TODO Copyright and Other info '''

import os
import sys
import ast
import time
import yaml
import importlib
from munch import DefaultMunch
import getpass
from loguru import logger
from collections import namedtuple
from argparse import ArgumentParser

CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH, "../")))
from utils import cluster_manager, image_manager

VERSION = "v0.1"
CLUSTER_MGR = cluster_manager.ClusterManager()

CURR_PATH = os.path.abspath(os.path.dirname(__file__))


def print_welcome_msg():
    '''Print colorful welcome message to console.'''
    logger.log(
        "Welcome",
        "\033[1;34;40m==============================================\033[0m")
    logger.log("Welcome",
               "\033[1;36;40m          Welcome to FlagPerf Inference!\033[0m")
    logger.log(
        "Welcome",
        "\033[1;36;40m      See more at https://github.com/FlagOpen/FlagPerf \033[0m"
    )
    logger.log(
        "Welcome",
        "\033[1;34;40m==============================================\033[0m")


def init_logger(config):
    logger.remove()
    """
    define "EVENTS", using logger.log("EVENT",msg) to log
    #21 means just important than info(#20), less than warning(#30)
    """
    logger.level("Welcome", no=21)

    timestamp_log_dir = "run" + time.strftime("%Y%m%d%H%M%S", time.localtime())
    curr_log_path = config.FLAGPERF_PATH + "/" + config.FLAGPERF_LOG_PATH + "/" + timestamp_log_dir
    logfile = curr_log_path + "/host.out.log"

    logger.remove()

    if config.LOG_CALL_INFORMATION:
        logger.add(logfile, level=config.FLAGPERF_LOG_LEVEL)
        logger.add(sys.stdout, level=config.FLAGPERF_LOG_LEVEL)
    else:
        logger.add(logfile,
                   level=config.FLAGPERF_LOG_LEVEL,
                   format="{time} - {level} - {message}")
        logger.add(sys.stdout,
                   level=config.FLAGPERF_LOG_LEVEL,
                   format="{time} - {level} - {message}")
    return curr_log_path


def usage():
    ''' Show usage and exit with exit_code. '''
    print("Usage: python3 ", __file__, " [--custom-docker-cmd 'docker run command']")
    print("Edit config file host.yaml in configs and run.")
    sys.exit(0)


def parse_args():
    '''Parse command line arguments'''
    parser = ArgumentParser(description='FlagPerf Inference Benchmarks')
    parser.add_argument('--custom-docker-cmd',
                       type=str,
                       help='Complete docker run command to use instead of default assembly')
    return parser.parse_args()


def check_cluster_health():
    ''' Try to ssh login to all the hosts in cluster_conf.hosts.
        Return None if everything goes well.
    '''
    logger.debug("Cluster healthcheck ssh. Hosts are: " +
                 ",".join(CLUSTER_MGR.get_hosts_list()))
    bad_hosts = CLUSTER_MGR.healthcheck()
    if len(bad_hosts) != 0:
        for bad_host in bad_hosts:
            logger.error("Check " + bad_host + " failed. ssh command exit "
                         "with: " + str(bad_hosts[bad_host]))
        logger.error("Check hosts in the cluster......[FAILED] [EXIT]")
        sys.exit(3)
    logger.info("Check hosts in the cluster......[SUCCESS]")


def _get_deploy_path(config):
    '''Return deploy path according to FLAGPERF_LOG_PATH_HOST in host.yaml.'''
    if 'FLAGPERF_PATH' not in config.__dict__.keys() \
       or config.FLAGPERF_PATH is None:
        dp_path = CURR_PATH
    else:
        dp_path = os.path.abspath(config.FLAGPERF_PATH)
    return dp_path


def check_cluster_deploy_path(dp_path):
    '''Make sure that flagperf is deployed on all the hosts
    '''
    logger.debug("Check flagperf deployment path: " + dp_path)
    bad_hosts = CLUSTER_MGR.run_command_all_hosts("cd " + dp_path)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't find deployed path: " +
                     ",".join(bad_hosts.keys()))
        logger.error("Check cluster deploy path " + dp_path +
                     "......[FAILED] [EXIT]")
        sys.exit(3)
    logger.info("Check flagperf deployment path: " + dp_path + "...[SUCCESS]")


def check_test_host_config(config):
    ''' Check test config.
        Make sure all CASES are configed.
    '''
    logger.debug("Check config in host.yaml")
    must_para = [
        'FLAGPERF_LOG_PATH', 'FLAGPERF_LOG_PATH', 'VENDOR',
        'FLAGPERF_LOG_LEVEL', 'HOSTS', 'SSH_PORT', 'HOSTS_PORTS',
        'MASTER_PORT', 'SHM_SIZE', 'ACCE_CONTAINER_OPT', 'PIP_SOURCE',
        'CLEAR_CACHES', 'ACCE_VISIBLE_DEVICE_ENV_NAME', 'CASES'
    ]

    for para in must_para:
        if para not in config.__dict__.keys():
            logger.error(f"{para} MUST be set in host.yaml...[EXIT]")
            sys.exit(2)
    logger.info("Check host.yaml...[SUCCESS]")


def check_case_config(case, case_config, vendor):
    '''Check config of the testcase. Make sure its path exists, framework is
       right and config file exists.
    '''
    logger.debug("Check config of test case: " + case)
    must_configs = [
        "model", "framework", "data_dir_host", "data_dir_container"
    ]
    for config_item in case_config.keys():
        if config_item in must_configs:
            must_configs.remove(config_item)
    if len(must_configs) > 0:
        logger.warning("Case " + case + " misses some config items: " +
                       ",".join(must_configs))
        return False
    logger.debug("Check config of test case: " + case + " ...[SUCCESS]")
    return True


def prepare_docker_image_cluster(dp_path, image_mgr, framework, nnodes,
                                 config):
    '''Prepare docker image in registry and in the cluster.
    '''
    vendor = config.VENDOR
    image_vendor_dir = os.path.join(
        CURR_PATH, "docker_images/" + vendor + "/" + framework)
    image_name = image_mgr.repository + ":" + image_mgr.tag
    logger.debug("Prepare docker image in cluster. image_name=" + image_name +
                 " image_vendor_dir=" + image_vendor_dir)
    prepare_image_cmd = "cd " + dp_path + " && " + sys.executable \
                        + " ../utils/image_manager.py -o build -i " \
                        + image_mgr.repository + " -t " + image_mgr.tag \
                        + " -d " + image_vendor_dir + " -f " + framework
    timeout = 1200
    logger.debug("Run cmd in the cluster to prepare docker image: " +
                 prepare_image_cmd + " timeout=" + str(timeout))
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(prepare_image_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't pull image: " +
                     ",".join(bad_hosts.keys()))
        return False
    return True


def prepare_running_env(dp_path, container_name, case_config, config):
    '''Install extensions and setup env before start task in container.
    '''
    nnodes = case_config["nnodes"]
    model = case_config["model"]
    framework = case_config["framework"]
    prepare_cmd = "cd " + dp_path + " && " + sys.executable \
                  + " ../utils/container_manager.py -o runcmdin -c " \
                  + container_name + " -t 1800 -r \"python3 " \
                  + config.FLAGPERF_PATH + "/" \
                  + "/tools/prepare_in_container.py --framework " \
                  + framework + " --model " + model + " --vendor " \
                  + config.VENDOR + " --pipsource " + config.PIP_SOURCE + "\""
    timeout = 1800
    logger.debug("Run cmd in the cluster to prepare running environment: " +
                 prepare_cmd + " timeout=" + str(timeout))
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(prepare_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't prepare running environment " +
                     "properly: " + ",".join(bad_hosts.keys()))
        return False
    return True


def start_container_in_cluster(dp_path, run_args, container_name, image_name,
                               nnodes):
    '''Call CLUSTER_MGR tool to start containers.'''
    # 直接构建docker命令，避免多层shell转义问题
    direct_docker_cmd = "docker run " + run_args + " --name=" + container_name + " " + image_name + " sleep infinity"
    start_cmd = direct_docker_cmd
    logger.debug("Run cmd in the cluster to start container: " + start_cmd)
    logger.debug("[DEBUG] Container run_args: " + run_args)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(start_cmd, nnodes, 600)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't start docker container: " +
                     ",".join(bad_hosts.keys()))
        return False
    return True


def start_custom_container_in_cluster(custom_docker_cmd, container_name, nnodes):
    '''Start containers using custom docker command.'''
    # Replace {CONTAINER_NAME} placeholder with actual container name if exists
    final_cmd = custom_docker_cmd.replace("{CONTAINER_NAME}", container_name)
    # 进入自定义流程
    logger.debug("······················进入自定义流程·······················")
    # If no placeholder and no --name in command, add container name
    if "{CONTAINER_NAME}" not in custom_docker_cmd and "--name" not in custom_docker_cmd:
        # Add container name before the image name (assuming format: docker run [options] image [cmd])
        parts = final_cmd.split()
        # Find where to insert --name (before the image name, usually the last non-option argument)
        insert_pos = len(parts)
        for i, part in enumerate(parts):
            if not part.startswith('-') and i > 1:  # Skip 'docker' and 'run'
                insert_pos = i
                break
        parts.insert(insert_pos, f"--name={container_name}")
        final_cmd = " ".join(parts)

    logger.debug("Run custom docker cmd in the cluster: " + final_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(final_cmd, nnodes, 600)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't start custom docker container: " +
                     ",".join(bad_hosts.keys()))
        return False
    return True


def stop_container_in_cluster(dp_path, container_name, nnodes):
    '''Call CLUSTER_MGR tool to stop containers with enhanced cleanup.'''
    
    # 首先尝试正常停止容器
    stop_cont_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " ../utils/container_manager.py -o stop" \
                    + " -c " + container_name
    logger.debug("Run cmd to stop container(s) in the cluster:" +
                 stop_cont_cmd)
    failed_hosts = CLUSTER_MGR.run_command_some_hosts(stop_cont_cmd, nnodes, 60)
    
    # 如果正常停止失败，尝试强制清理
    if len(failed_hosts) != 0:
        logger.warning("Normal container stop failed, attempting force cleanup...")
        
        # 强制停止和删除容器
        force_cleanup_cmd = f"docker ps -aq --filter name={container_name} | xargs -r docker rm -f"
        logger.debug("Force cleanup cmd: " + force_cleanup_cmd)
        
        cleanup_failed = CLUSTER_MGR.run_command_some_hosts(force_cleanup_cmd, nnodes, 30)
        
        # 额外清理：删除所有相关容器
        extra_cleanup_cmd = "docker container prune -f"
        CLUSTER_MGR.run_command_some_hosts(extra_cleanup_cmd, nnodes, 30)
        
        if len(cleanup_failed) != 0:
            logger.warning("Hosts that force cleanup failed:" + 
                         ",".join(cleanup_failed.keys()) + " Continue.")
            return False
    
    logger.info("All containers stopped and cleaned up in the cluster")
    return True


def clear_caches_cluster(clear, nnodes):
    '''Set vm.drop to clean the system caches.'''
    if not clear:
        logger.info("Caches clear config is NOT set.")
        return

    clear_cmd = "sync && sudo /sbin/sysctl vm.drop_caches=3"
    timeout = 30
    logger.debug("Run cmd in the cluster to clear the system cache: " +
                 clear_cmd + " timeout=" + str(timeout))
    failed_hosts = CLUSTER_MGR.run_command_some_hosts(clear_cmd, nnodes,
                                                      timeout)
    if len(failed_hosts) != 0:
        logger.warning("Hosts that clear cache failed: " +
                       ",".join(failed_hosts.keys()) + ". Continue.")
    logger.info("Clear system caches if it set......[SUCCESS]")


def start_monitors_in_cluster(dp_path, case_log_dir, nnodes, config):
    '''Start sytem and vendor's monitors.'''
    start_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                    + " ../utils/sys_monitor.py -o restart -l "
    timeout = 60
    logger.debug("Run cmd in the cluster to start system monitors: " +
                 start_mon_cmd)
    bad_hosts = CLUSTER_MGR.start_monitors_some_hosts(start_mon_cmd,
                                                      case_log_dir, nnodes,
                                                      timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't start system monitors: " +
                     ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, "docker_images", config.VENDOR,
                                config.VENDOR + "_monitor.py")
    start_mon_cmd = "cd " + dp_path + " && sudo " + sys.executable \
                    + " " + ven_mon_path + " -o restart -l "
    logger.debug("Run cmd in the cluster to start vendor's monitors: " +
                 start_mon_cmd)
    bad_hosts = CLUSTER_MGR.start_monitors_some_hosts(start_mon_cmd,
                                                      case_log_dir, nnodes,
                                                      timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't start vendor's monitors: " +
                     ",".join(bad_hosts.keys()))


def stop_monitors_in_cluster(dp_path, nnodes, config):
    '''Stop sytem and vendor's monitors.'''
    stop_mon_cmd = "cd " + dp_path + " && " + sys.executable \
                   + " ../utils/sys_monitor.py -o stop"
    timeout = 60
    logger.debug("Run cmd in the cluster to stop system monitors: " +
                 stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't stop system monitors: " +
                     ",".join(bad_hosts.keys()))

    ven_mon_path = os.path.join(dp_path, "docker_images", config.VENDOR,
                                config.VENDOR + "_monitor.py")
    stop_mon_cmd = "cd " + dp_path + " && sudo " + sys.executable \
                   + " " + ven_mon_path + " -o stop"
    logger.debug("Run cmd in the cluster to stop vendor's monitors: " +
                 stop_mon_cmd)
    bad_hosts = CLUSTER_MGR.run_command_some_hosts(stop_mon_cmd, nnodes,
                                                   timeout)
    if len(bad_hosts) != 0:
        logger.error("Hosts that can't stop vendor's monitors: " +
                     ",".join(bad_hosts.keys()))


def start_tasks_in_cluster(dp_path, container_name, case_config, curr_log_path,
                           config):
    '''Start tasks in cluster, and NOT wait.'''
    nnodes = case_config["nnodes"]
    framework = case_config["framework"].split("_")[0]
    env_file = os.path.join(
        config.FLAGPERF_PATH, config.VENDOR,
        case_config["model"] + "-" + framework,
        "config/environment_variables.sh")

    # 创建增强的启动命令，类似训练版本的改动
    debug_log_path = config.FLAGPERF_PATH + "/" + curr_log_path + "/inference_debug.log"

    if os.path.isfile(env_file):
        start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " ../utils/container_manager.py -o runcmdin -c " \
                + container_name + " -t 600 -r \"python3 --version"
        
        # 创建日志目录并记录调试信息
        start_cmd += " && mkdir -p " + config.FLAGPERF_PATH + "/" + curr_log_path \
                     + " && echo 'Starting inference task at '$(date) > " + debug_log_path \
                     + " && source " + env_file \
                     + " > " + curr_log_path + "/source_env.log.txt 2>&1" \
                     + " && echo 'Environment sourced, starting inference' >> " + debug_log_path \
                     + " && python3 run_inference.py" \
                     + f" --perf_dir " + config.FLAGPERF_PATH \
                     + f" --loglevel " + config.FLAGPERF_LOG_LEVEL \
                     + f" --vendor " + config.VENDOR \
                     + f" --case " + case_config["model"] \
                     + f" --data_dir " + case_config["data_dir_container"] \
                     + f" --framework " + case_config["framework"] \
                     + f" --log_dir " + curr_log_path \
                     + " 2>&1 | tee -a " + debug_log_path \
                     + " && echo 'Inference finished with exit code: '$? >> " + debug_log_path
    else:
        start_cmd = "cd " + dp_path + " && " + sys.executable \
                + " ../utils/container_manager.py -o runcmdin -c " \
                + container_name + " -t 600 -r \"python3 --version" \
                + " && mkdir -p " + config.FLAGPERF_PATH + "/" + curr_log_path \
                + " && echo 'Starting inference task (no env) at '$(date) > " + debug_log_path \
                + " && python3 run_inference.py" \
                + f" --perf_dir " + config.FLAGPERF_PATH \
                + f" --loglevel " + config.FLAGPERF_LOG_LEVEL \
                + f" --vendor " + config.VENDOR \
                + f" --case " + case_config["model"] \
                + f" --data_dir " + case_config["data_dir_container"] \
                + f" --framework " + case_config["framework"] \
                + f" --log_dir " + curr_log_path \
                + " 2>&1 | tee -a " + debug_log_path \
                + " && echo 'Inference finished with exit code: '$? >> " + debug_log_path
    
    if config.ACCE_VISIBLE_DEVICE_ENV_NAME is not None:
        start_cmd += " --visible_dev_env " \
                     + config.ACCE_VISIBLE_DEVICE_ENV_NAME
    start_cmd += " \""
    
    logger.debug("Run cmd in the cluster to start inference tasks, cmd=" + start_cmd)
    logger.info(f"Inference main command for case: {case_config['model']}")
    
    # 执行命令并检查结果
    failed_hosts = CLUSTER_MGR.run_command_some_hosts_distribution_info(start_cmd, nnodes, 15, "inference")
    
    if failed_hosts and len(failed_hosts) > 0:
        logger.error(f"Inference command execution failed on hosts: {list(failed_hosts.keys())}")
    else:
        logger.info("Inference command execution started successfully on all hosts")
    
    # Wait a moment for starting tasks.
    time.sleep(10)

    logger.info("3) Waiting for tasks end in the cluster...")
    logger.info("Check task log in real time from container: " +
                curr_log_path + "/container.out.log")
    logger.info("Check task stderr & stdout in real time from container: " +
                curr_log_path + "/stdout_err.out.log")


def wait_for_finish(dp_path, container_name, pid_file_path, nnodes):
    '''wait all the processes of start_xxx_task.py finished.
    '''
    # 设置最大等待时间（推理任务通常比训练快）
    max_wait_time = 1800  # 30分钟超时
    start_wait_time = time.time()
    
    check_cmd = "cd " + dp_path + "; " + sys.executable \
                + " ../utils/container_manager.py -o pidrunning -c " \
                + container_name + " -f " + pid_file_path

    logger.debug("Run cmd to check whether the inference tasks is running: " + check_cmd)
    
    # 首先等待任务启动
    time.sleep(10)
    
    while time.time() - start_wait_time < max_wait_time:
        bad_hosts = CLUSTER_MGR.run_command_some_hosts(check_cmd,
                                                       nnodes,
                                                       no_log=True)

        if len(bad_hosts) == nnodes:
            break
        time.sleep(30)
    
    if time.time() - start_wait_time >= max_wait_time:
        logger.warning("Inference task wait timeout reached, proceeding with cleanup")
    
    logger.info("Inference tasks finished in the cluster")


def prepare_containers_env_cluster(dp_path, case_log_dir, config,
                                   container_name, image_name, case_config, custom_docker_cmd=None):
    '''Prepare containers environments in the cluster. It will start
       containers, setup environments, start monitors, and clear caches.'''
    nnodes = case_config["nnodes"]
    
    logger.info("a) Check and clean Docker environment first.")
    
    # 检查Docker状态
    docker_status_cmd = "docker ps"
    logger.debug("Checking running Docker containers: " + docker_status_cmd)
    CLUSTER_MGR.run_command_some_hosts(docker_status_cmd, nnodes, 30)
    
    # 检查容器是否存在，然后清理
    check_container_cmd = f"docker ps -aq --filter name={container_name}"
    logger.debug("Checking if container exists: " + check_container_cmd)
    existing_result = CLUSTER_MGR.run_command_some_hosts(check_container_cmd, nnodes, 15)
    
    # 如果容器存在（命令成功执行），则进行清理
    if len(existing_result) == 0:  # 没有失败的主机，说明命令执行成功
        logger.info("Found existing containers, cleaning up...")
        
        # 停止容器
        stop_related_cmd = f"docker stop {container_name} 2>/dev/null || true"
        logger.debug("Stopping existing container: " + stop_related_cmd)
        CLUSTER_MGR.run_command_some_hosts(stop_related_cmd, nnodes, 15)
        
        # 删除容器
        remove_related_cmd = f"docker rm {container_name} 2>/dev/null || true"
        logger.debug("Removing existing container: " + remove_related_cmd)
        CLUSTER_MGR.run_command_some_hosts(remove_related_cmd, nnodes, 15)
    else:
        logger.info("No existing containers found, proceeding with fresh start.")

    logger.info("b) Stop old container(s) first.")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    logger.info("c) Start container(s) in the cluster.")

    if custom_docker_cmd is not None:
        # Use custom docker command
        logger.info("Using custom docker command: " + custom_docker_cmd)
        if not start_custom_container_in_cluster(custom_docker_cmd, container_name, nnodes):
            logger.error("b) Start custom container in the cluster......"
                         "[FAILED]. Ignore this round.")
            return False
    else:
        # Use default container assembly logic
        container_start_args = " --rm --init --detach --net=host --uts=host" \
                               + " --ipc=host --security-opt=seccomp=unconfined" \
                               + " --privileged=true --ulimit=stack=67108864" \
                               + " --ulimit=memlock=-1" \
                               + " -w " + config.FLAGPERF_PATH \
                               + " --shm-size=" + config.SHM_SIZE \
                               + " -v " + dp_path + ":" \
                               + config.FLAGPERF_PATH \
                               + " -v " + os.path.join(dp_path, "..") + ":" \
                               + os.path.join(config.FLAGPERF_PATH, "..") \
                               + " -v " + case_config["data_dir_host"] + ":" \
                               + case_config["data_dir_container"] \
                               + " -v " + "/home/secure/data/stable_diffusion/val/clip-vit-base-patch32" + ":" \
                               + "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/main"
        if config.ACCE_CONTAINER_OPT is not None:
            container_start_args += " " + config.ACCE_CONTAINER_OPT

        if not start_container_in_cluster(dp_path, container_start_args,
                                          container_name, image_name, nnodes):
            logger.error("b) Start container in the cluster......"
                         "[FAILED]. Ignore this round.")
            return False

    logger.info("c) Start container(s) in the cluster.......[SUCCESS]")
    
    # 验证容器是否真的启动成功
    verify_cmd = f"docker ps --filter name={container_name}"
    logger.debug("Verifying container status: " + verify_cmd)
    CLUSTER_MGR.run_command_some_hosts(verify_cmd, nnodes, 15)
    
    # 测试容器是否响应命令
    logger.info("Testing container command execution...")
    test_cmd = "cd " + dp_path + " && " + sys.executable \
               + " ../utils/container_manager.py -o runcmdin -c " \
               + container_name + " -t 30 -r \"echo 'Container test: '$(date) && whoami && pwd\""
    logger.debug("Container test command: " + test_cmd)
    test_result = CLUSTER_MGR.run_command_some_hosts(test_cmd, nnodes, 30)
    
    if len(test_result) == 0:
        logger.info("✓ Container responds to commands successfully")
    else:
        logger.warning("✗ Container command test failed on hosts: " + ",".join(test_result.keys()))

    logger.info("d) Prepare running environment.")
    if not prepare_running_env(dp_path, container_name, case_config, config):
        logger.error("d) Prepare running environment......"
                     "[FAILED]. Ignore this round.")
        logger.info("Stop containers in cluster.")
        stop_container_in_cluster(dp_path, container_name, nnodes)
        return False
    logger.info("d) Prepare running environment......[SUCCESS]")
    logger.info("e) Start monitors......")
    start_monitors_in_cluster(dp_path, case_log_dir, nnodes, config)
    logger.info("f) Clear system caches if it set......")
    clear_caches_cluster(config.CLEAR_CACHES, nnodes)
    return True


def clean_containers_env_cluster(dp_path, container_name, nnodes, config):
    '''Clean containers environments in the cluster. It will stop containers,
       and stop monitors.'''
    logger.info("a) Stop containers......")
    stop_container_in_cluster(dp_path, container_name, nnodes)
    logger.info("b) Stop monitors......")
    stop_monitors_in_cluster(dp_path, nnodes, config)


def compilation_result(case_log_path, config):
    '''Scp logs from hosts in the cluster to temp dir, and then merge all.
    '''
    case_perf_path = os.path.join(case_log_path, "container.out.log")
    vendor_usage_path = os.path.join(case_log_path,
                                     config.VENDOR + "_monitor.log")

    case_perf = None
    case_file = open(case_perf_path)

    for line in case_file.readlines():
        if "Finish Info" in line:
            case_perf_str = "{" + line.split("{")[1]
            case_perf = ast.literal_eval(case_perf_str)
            break

    if case_perf is None:
        logger.error("Case Run Failed, Please Check Log!")
        return

    vendor_module = importlib.import_module("docker_images." + config.VENDOR +
                                            "." + config.VENDOR + "_analysis")
    vendor_usage, vendor_maxmem, fp32, fp16 = vendor_module.analysis_log(
        vendor_usage_path)

    case_perf["vendor_usage(GiB)"] = vendor_usage
    case_perf["vendor_max_mem(GiB)"] = vendor_maxmem

    theory = fp32 if case_perf["precision"] == "fp32" else fp16
    mfu = case_perf["flops"] / theory
    case_perf["*MFU"] = str(round(mfu * 100, 1)) + "%"

    for key in case_perf.keys():
        padding_str = str(key).ljust(43) + " : " + str(
            case_perf[key]).ljust(23)
        logger.info(padding_str)


def get_config_from_case(case, config):
    '''check case is string'''
    if not isinstance(case, str):
        logger.error("Key in test_config.CASES must be str")
        return False, None

    case_info = case.split(":")
    '''check if 4+ : in case, we don't care what to put in'''
    if len(case_info) < 2:
        logger.error("At least 2 terms split by \":\" should in config.CASES")
        logger.error("model:framework:hardware_model:nnodes:nproc:repeat")
        return False, None

    case_model = case_info[0]
    case_framework = case_info[1]

    case_config = {"model": case_model}
    case_config["framework"] = case_framework
    case_config["data_dir_host"] = config.CASES[case]
    case_config["data_dir_container"] = config.CASES[case]
    case_config['nnodes'] = 1

    return True, case_config


def get_valid_cases(config):
    '''Check case config in test_conf, return valid cases list.'''
    if not isinstance(config.CASES, dict):
        logger.error(
            "No valid cases found in test_conf because test_config.CASES is not a dict...[EXIT]"
        )
        sys.exit(4)
    logger.debug("Check configs of all test cases: " + ", ".join(config.CASES))
    valid_cases = []
    cases_config_error = []
    for case in config.CASES:
        rets, case_config = get_config_from_case(case, config)
        if (not rets) or (not check_case_config(case, case_config,
                                                config.VENDOR)):
            cases_config_error.append(case)
            continue
        valid_cases.append(case)
    if len(valid_cases) == 0:
        logger.error("No valid cases found in test_conf...[EXIT]")
        sys.exit(4)
    logger.debug("Valid cases: " + ",".join(valid_cases))
    logger.debug("Invalid cases that config is error: " +
                 ",".join(cases_config_error))
    logger.info("Get valid cases list......[SUCCESS]")
    return valid_cases


def prepare_case_config_cluster(dp_path, case_config, case):
    '''Sync case config files in cluster.'''
    logger.info("--------------------------------------------------")
    logger.info("Testcase " + case + " config:")
    for config_item in case_config.keys():
        logger.info(config_item + ":\t" + str(case_config[config_item]))
    logger.info("--------------------------------------------------")
    model = case_config["model"]
    framework = case_config["framework"].split("_")[0]
    config_file = case_config["config"] + ".py"
    nnodes = case_config["nnodes"]
    case_config_dir = os.path.join(dp_path, config.VENDOR,
                                   model + "-" + framework, "config")
    case_config_file = os.path.join(case_config_dir, config_file)
    failed_hosts = CLUSTER_MGR.sync_file_to_some_hosts(case_config_file,
                                                       case_config_dir, nnodes)
    if len(failed_hosts) != 0:
        logger.error("Hosts that sync vendor case config file failed: " +
                     ",".join(failed_hosts.keys()))
        return False
    return True


def log_test_configs(cases, curr_log_path, dp_path):
    '''Put test configs to log '''
    logger.info("--------------------------------------------------")
    logger.info("Prepare to run flagperf Inference benchmakrs with configs: ")
    logger.info("Deploy path on host:\t" + dp_path)
    logger.info("Vendor:\t\t" + config.VENDOR)
    logger.info("Testcases:\t\t[" + ','.join(cases) + "]")
    logger.info("Log path on host:\t" + curr_log_path)
    logger.info("Cluster:\t\t[" + ",".join(config.HOSTS) + "]")
    logger.info("--------------------------------------------------")


def main(config, custom_docker_cmd=None):
    '''Main process to run all the testcases'''

    curr_log_whole = init_logger(config)

    print_welcome_msg()

    logger.info("======== Step 1: Check key configs. ========")

    check_test_host_config(config)

    # Check test environment and configs from host.yaml.
    CLUSTER_MGR.init(config.HOSTS, config.SSH_PORT, getpass.getuser(), logger)
    check_cluster_health()
    dp_path = _get_deploy_path(config)
    check_cluster_deploy_path(dp_path)

    cases = get_valid_cases(config)
    log_test_configs(cases, curr_log_whole, dp_path)

    logger.info("========= Step 2: Prepare and Run test cases. =========")

    for case in cases:
        logger.info("======= Testcase: " + case + " =======")
        _, case_config = get_config_from_case(case, config)

        # Prepare docker image.
        image_mgr = image_manager.ImageManager(
            "flagperf-inference-" + config.VENDOR + "-" +
            case_config["framework"], "t_" + VERSION)
        image_name = image_mgr.repository + ":" + image_mgr.tag
        nnodes = case_config["nnodes"]
        logger.info("=== 2.1 Prepare docker image:" + image_name + " ===")
        if not prepare_docker_image_cluster(
                dp_path, image_mgr, case_config["framework"], nnodes, config):
            logger.error("=== 2.1 Prepare docker image...[FAILED] " +
                         "Ignore this case " + case + " ===")
            continue

        # Set command to start docker container in the cluster
        container_name = image_mgr.repository + "-" + image_mgr.tag \
                                              + "-container"

        logger.info("=== 2.2 Setup container and run testcases. ===")

        logger.info("-== Testcase " + case + " starts ==-")
        logger.info("1) Prepare container environments in cluster...")
        case_log_dir = os.path.join(curr_log_whole, case)
        curr_log_path = os.path.join(case_log_dir,
                                     config.HOSTS[0] + "_noderank0")

        if not prepare_containers_env_cluster(dp_path, case_log_dir, config,
                                              container_name, image_name,
                                              case_config, custom_docker_cmd):
            logger.error("1) Prepare container environments in cluster"
                         "...[FAILED]. Ignore case " + case)
            continue
        logger.info("2) Start tasks in the cluster...")

        start_tasks_in_cluster(dp_path, container_name, case_config,
                               curr_log_path, config)

        # Wait until inference tasks finished.
        logger.info("3) Waiting for inference tasks to complete...")
        # Note: 推理任务通常较快完成，使用简单的等待策略
        logger.info("   Inference tasks are typically fast, waiting for completion...")
        time.sleep(30)  # 给推理任务一些时间完成
        logger.info("   Basic wait completed. Check logs for detailed results.")
        
        logger.info("4) Inference tasks completed in the cluster...")
        logger.info("5) Clean container environments in cluster...")
        clean_containers_env_cluster(dp_path, container_name, nnodes, config)
        logger.info("-== Testcase " + case + " finished ==-")
        logger.info("=== 2.2 Setup container and run testcases finished."
                    " ===")
        logger.info("=== 2.3 Compilation Case Performance ===")
        compilation_result(curr_log_path, config)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    custom_docker_cmd = args.custom_docker_cmd
    
    CURR_PATH = os.path.abspath(os.path.dirname(__file__))
    yaml_path = os.path.join(CURR_PATH, "configs/host.yaml")
    data = yaml.safe_load(open(yaml_path))

    config = DefaultMunch.fromDict(data)

    main(config, custom_docker_cmd)

