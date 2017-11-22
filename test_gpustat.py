from __future__ import print_function

import copy
import unittest

import psutil
import pynvml

import polyaxon_gpustat

from collections import namedtuple

try:
    import unittest.mock as mock
except:
    import mock

MagicMock = mock.MagicMock


def _configure_mock(N, Process, scenario_nonexistent_pid=False):
    """
    Define mock behaviour for N: the pynvml module, and psutil.Process,
    which should be MagicMock objects from unittest.mock.
    """

    # Restore some non-mock objects (such as exceptions)
    for attr in dir(pynvml):
        if attr.startswith('NVML'):
            setattr(N, attr, getattr(pynvml, attr))
    assert issubclass(N.NVMLError, BaseException)

    # without following patch, unhashable NVMLError distrubs unit test
    N.NVMLError.__hash__ = lambda _: 0

    # mock-patch every nvml**** functions used in gpustat.
    N.nvmlInit = MagicMock()
    N.nvmlShutdown = MagicMock()
    N.nvmlDeviceGetCount.return_value = 3

    mock_handles = ['mock-handle-%d' % i for i in range(3)]

    def _raise_ex(fn):
        """ Decorator to let exceptions returned from the callable re-throwed. """

        def _decorated(*args, **kwargs):
            v = fn(*args, **kwargs)
            if isinstance(v, Exception): raise v
            return v

        return _decorated

    device = namedtuple("Device", ['busId'])
    N.nvmlDeviceGetPciInfo.side_effect = (lambda index: device(busId="0000:00:1E.1"))
    N.nvmlDeviceGetSerial.side_effect = (lambda index: '0322917092147')
    N.nvmlDeviceGetHandleByIndex.side_effect = (lambda index: mock_handles[index])
    N.nvmlDeviceGetName.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: b'GeForce GTX TITAN 0',
        mock_handles[1]: b'GeForce GTX TITAN 1',
        mock_handles[2]: b'GeForce GTX TITAN 2',
    }.get(handle, RuntimeError))
    N.nvmlDeviceGetUUID.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: b'GPU-10fb0fbd-2696-43f3-467f-d280d906a107',
        mock_handles[1]: b'GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2',
        mock_handles[2]: b'GPU-50205d95-57b6-f541-2bcb-86c09afed564',
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetTemperature = _raise_ex(lambda handle, _: {
        mock_handles[0]: 80,
        mock_handles[1]: 36,
        mock_handles[2]: 71,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetPowerUsage = _raise_ex(lambda handle: {
        mock_handles[0]: 125000,
        mock_handles[1]: 100000,
        mock_handles[2]: 250000,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetEnforcedPowerLimit = _raise_ex(lambda handle: {
        mock_handles[0]: 250000,
        mock_handles[1]: 250000,
        mock_handles[2]: 250000,
    }.get(handle, RuntimeError))

    mock_memory_t = namedtuple("Memory_t", ['total', 'used', 'free'])
    N.nvmlDeviceGetMemoryInfo.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: mock_memory_t(total=12883853312, used=8000 * MB, free=1000),
        mock_handles[1]: mock_memory_t(total=12781551616, used=9000 * MB, free=1000),
        mock_handles[2]: mock_memory_t(total=12781551616, used=0, free=12781551616),
    }.get(handle, RuntimeError))

    mock_utilization_t = namedtuple("Utilization_t", ['gpu', 'memory'])
    N.nvmlDeviceGetUtilizationRates.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: mock_utilization_t(gpu=76, memory=0),
        mock_handles[1]: mock_utilization_t(gpu=0, memory=0),
        mock_handles[2]: N.NVMLError_NotSupported(),  # Not Supported
    }.get(handle, RuntimeError))

    # running process information: a bit annoying...
    mock_process_t = namedtuple("Process_t", ['pid', 'usedGpuMemory'])

    if scenario_nonexistent_pid:
        mock_processes_gpu2_erratic = [mock_process_t(99999, 9999 * MB)]
    else:
        mock_processes_gpu2_erratic = N.NVMLError_NotSupported()
    N.nvmlDeviceGetComputeRunningProcesses.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: [mock_process_t(48448, 4000 * MB), mock_process_t(153223, 4000 * MB)],
        mock_handles[1]: [mock_process_t(192453, 3000 * MB), mock_process_t(194826, 6000 * MB)],
        mock_handles[2]: mock_processes_gpu2_erratic,  # Not Supported or non-existent
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetGraphicsRunningProcesses.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: [],
        mock_handles[1]: [],
        mock_handles[2]: N.NVMLError_NotSupported(),
    }.get(handle, RuntimeError))

    mock_pid_map = {  # mock information for psutil...
        48448: ('user1', 'python'),
        154213: ('user1', 'caffe'),
        38310: ('user3', 'python'),
        153223: ('user2', 'python'),
        194826: ('user3', 'caffe'),
        192453: ('user1', 'torch'),
    }

    def _MockedProcess(pid):
        if not pid in mock_pid_map:
            raise psutil.NoSuchProcess(pid=pid)
        username, cmdline = mock_pid_map[pid]
        p = MagicMock()  # mocked process
        p.username.return_value = username
        p.cmdline.return_value = [cmdline]
        return p

    Process.side_effect = _MockedProcess


MOCK_EXPECTED_OUTPUT_DEFAULT = """\
[0] GeForce GTX TITAN 0 | 80'C,  76 % |  8000 / 12287 MB | user1(4000M) user2(4000M)
[1] GeForce GTX TITAN 1 | 36'C,   0 % |  9000 / 12189 MB | user1(3000M) user3(6000M)
[2] GeForce GTX TITAN 2 | 71'C,  ?? % |     0 / 12189 MB | (Not Supported)
"""

MOCK_EXPECTED_OUTPUT_FULL = [
    {
        'index': 0,
        'bus_id': '0000:00:1E.1',
        'memory_free': 1000,
        'memory_total': 12883853312,
        'memory_used': 8388608000,
        'memory_utilization': 0,
        'minor': 1,
        'name': 'GeForce GTX TITAN 0',
        'power_draw': 125,
        'power_limit': 250,
        'processes': [{'command': 'python',
                       'gpu_memory_usage': 4000,
                       'pid': 48448,
                       'username': 'user1'},
                      {'command': 'python',
                       'gpu_memory_usage': 4000,
                       'pid': 153223,
                       'username': 'user2'}],
        'serial': '0322917092147',
        'temperature_gpu': 80,
        'utilization_gpu': 76,
        'uuid': 'GPU-10fb0fbd-2696-43f3-467f-d280d906a107'
    },
    {
        'bus_id': '0000:00:1E.1',
        'index': 1,
        'memory_free': 1000,
        'memory_total': 12781551616,
        'memory_used': 9437184000,
        'memory_utilization': 0,
        'minor': 1,
        'name': 'GeForce GTX TITAN 1',
        'power_draw': 100,
        'power_limit': 250,
        'processes': [{'command': 'torch',
                       'gpu_memory_usage': 3000,
                       'pid': 192453,
                       'username': 'user1'},
                      {'command': 'caffe',
                       'gpu_memory_usage': 6000,
                       'pid': 194826,
                       'username': 'user3'}],
        'serial': '0322917092147',
        'temperature_gpu': 36,
        'utilization_gpu': 0,
        'uuid': 'GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2'},
    {
        'bus_id': '0000:00:1E.1',
        'index': 2,
        'memory_free': 12781551616,
        'memory_total': 12781551616,
        'memory_used': 0,
        'memory_utilization': None,
        'minor': 1,
        'name': 'GeForce GTX TITAN 2',
        'power_draw': 250,
        'power_limit': 250,
        'processes': [],
        'serial': '0322917092147',
        'temperature_gpu': 71,
        'utilization_gpu': None,
        'uuid': 'GPU-50205d95-57b6-f541-2bcb-86c09afed564'
    }]

MB = 1024 * 1024


class TestGPUStat(unittest.TestCase):
    @mock.patch('psutil.Process')
    @mock.patch('polyaxon_gpustat.N')
    def test_query_mocked(self, N, Process):
        """
        A basic functionality test, in a case where everything is just normal.
        """
        _configure_mock(N, Process)

        gpustats = polyaxon_gpustat.query()
        self.assertEqual(gpustats[0], MOCK_EXPECTED_OUTPUT_FULL[0])
        self.assertEqual(gpustats[1], MOCK_EXPECTED_OUTPUT_FULL[1])
        self.assertEqual(gpustats[2].pop('processes'), None)
        mock_stats = copy.copy(MOCK_EXPECTED_OUTPUT_FULL[2])
        mock_stats.pop('processes')
        self.assertEqual(gpustats[2], mock_stats)

    @mock.patch('psutil.Process')
    @mock.patch('polyaxon_gpustat.N')
    def test_query_mocked_nonexistent_pid(self, N, Process):
        """
        Test a case where nvidia query returns non-existent pids (see #16, #18)
        """
        _configure_mock(N, Process, scenario_nonexistent_pid=True)

        gpustats = polyaxon_gpustat.query()
        self.assertEqual(gpustats, MOCK_EXPECTED_OUTPUT_FULL)
