"""Guard NVML host PIDs without equating them with a container os.getpid()."""
import os
import subprocess
import sys


def nvml_pids():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    return [int(line.strip()) for line in value.splitlines() if line.strip().isdigit()]


def context_initialized():
    torch = sys.modules.get('torch')
    return torch is not None and torch.cuda.is_initialized()


class OwnedCudaGuard:
    def __init__(self, scan=nvml_pids, initialized=context_initialized, on_bind=None):
        self.scan, self.initialized, self.on_bind = scan, initialized, on_bind
        self.saw_empty_before_cuda = False
        self.owned_nvml_pid = None

    def check(self):
        current = self.scan()
        if self.owned_nvml_pid is None:
            if not self.initialized():
                if current:
                    raise RuntimeError('Existing GPU users preserved: ' + str(current))
                self.saw_empty_before_cuda = True
                return
            # We own a real initialized CUDA context. A prior empty GPU under the
            # project lease plus exactly one current context identifies its NVML
            # host PID even when /proc in this container cannot see host PIDs.
            if not self.saw_empty_before_cuda or len(current) != 1:
                raise RuntimeError('Cannot uniquely bind owned CUDA context; preserve others: ' + str(current))
            self.owned_nvml_pid = current[0]
            if self.on_bind is not None:
                self.on_bind({'container_pid': os.getpid(), 'nvml_owned_context_pid': current[0],
                              'pid_namespaces_differ': current[0] != os.getpid(),
                              'binding': 'empty before context under project lease; exactly one after owned CUDA initialization'})
        others = [pid for pid in current if pid != self.owned_nvml_pid]
        if others or self.owned_nvml_pid not in current:
            raise RuntimeError('Foreign/missing GPU context; no queue or preemption: ' + str(current))
