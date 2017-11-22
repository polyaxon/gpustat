# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging

import psutil
import os.path

logger = logging.getLogger(__name__)

try:
    import pynvml as N
except ImportError:
    logger.debug("Could not import pynvml.  NVIDIA stats will not be collected.")
    has_gpu_nvidia = False
else:
    has_gpu_nvidia = True

NOT_SUPPORTED = 'Not Supported'


def query():
    """Query the information of all the GPUs on local machine"""

    N.nvmlInit()

    def get_gpu_info(handle):
        """Get one GPU information specified by nvml handle"""

        def get_process_info(pid):
            """Get the process information of specific pid"""
            process = {}
            ps_process = psutil.Process(pid=pid)
            process['username'] = ps_process.username()
            # cmdline returns full path; as in `ps -o comm`, get short cmdnames.
            _cmdline = ps_process.cmdline()
            if not _cmdline:  # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                process['command'] = '?'
            else:
                process['command'] = os.path.basename(_cmdline[0])
            # Bytes to MBytes
            process['gpu_memory_usage'] = int(nv_process.usedGpuMemory / 1024 / 1024)
            process['pid'] = nv_process.pid
            return process

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode()  # for python3, to unicode
            return b

        name = _decode(N.nvmlDeviceGetName(handle))
        uuid = _decode(N.nvmlDeviceGetUUID(handle))

        try:
            minor = int(N.nvmlDeviceGetMinorNumber(handle))
        except N.NVMLError:
            minor = None  # Not supported

        try:
            bus_id = _decode(N.nvmlDeviceGetPciInfo(handle).busId)
        except N.NVMLError:
            bus_id = None  # Not supported

        try:
            serial = _decode(N.nvmlDeviceGetSerial(handle))
        except N.NVMLError:
            serial = None  # Not supported

        try:
            temperature = N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)
        except N.NVMLError:
            temperature = None  # Not supported

        try:
            memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
        except N.NVMLError:
            memory = None  # Not supported

        try:
            utilization = N.nvmlDeviceGetUtilizationRates(handle)
        except N.NVMLError:
            utilization = None  # Not supported

        try:
            power = N.nvmlDeviceGetPowerUsage(handle)
        except (N.NVMLError, N.NVMLError_NotSupported):
            power = None

        try:
            power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
        except (N.NVMLError, N.NVMLError_NotSupported):
            power_limit = None

        processes = []
        try:
            nv_comp_processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
        except N.NVMLError:
            nv_comp_processes = None  # Not supported
        try:
            nv_graphics_processes = N.nvmlDeviceGetGraphicsRunningProcesses(handle)
        except N.NVMLError:
            nv_graphics_processes = None  # Not supported

        if nv_comp_processes is None and nv_graphics_processes is None:
            processes = None  # Not supported (in both cases)
        else:
            nv_comp_processes = nv_comp_processes or []
            nv_graphics_processes = nv_graphics_processes or []
            for nv_process in (nv_comp_processes + nv_graphics_processes):
                # TODO: could be more information such as system memory usage,
                # CPU percentage, create time etc.
                try:
                    process = get_process_info(nv_process.pid)
                    processes.append(process)
                except psutil.NoSuchProcess:
                    # TODO: add some reminder for NVML broken context
                    # e.g. nvidia-smi reset  or  reboot the system
                    pass

        gpu_info = {
            'index': index,
            'uuid': uuid,
            'name': name,
            'minor': minor,
            'bus_id': bus_id,
            'serial': serial,
            'temperature_gpu': temperature,
            'utilization_gpu': utilization.gpu if utilization else None,
            'power_draw': int(power / 1000) if power is not None else None,
            'power_limit': int(power_limit / 1000) if power is not None else None,
            'memory_free': int(memory.free) if memory else None,
            'memory_used': int(memory.used) if memory else None,
            'memory_total': int(memory.total) if memory else None,
            'memory_utilization': utilization.memory if utilization else None,
            'processes': processes,
        }
        return gpu_info

    # 1. get the list of gpu and status
    gpu_list = []
    device_count = N.nvmlDeviceGetCount()

    for index in range(device_count):
        handle = N.nvmlDeviceGetHandleByIndex(index)
        gpu_info = get_gpu_info(handle)
        gpu_list.append(gpu_info)

    N.nvmlShutdown()
    return gpu_list
