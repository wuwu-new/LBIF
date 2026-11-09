"""One finite CPU-only ABBA trial; no torch import or CUDA context."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import plan as p
from prefetch import ordered_batches, target_arrays
from checked_pixels import pixels


def main():
    os.nice(19)
    run=Path(p.REMOTE)/p.RUN
    output=run/'CPU_BENCHMARK.json'
    assert not output.exists() and not (run/'CPU_BENCHMARK_STARTED.json').exists()
    def dump(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2)
    lock=json.loads((run/'REGISTRATION.json').read_text())
    for name,digest in lock['files'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
    dump(run/'CPU_BENCHMARK_STARTED.json',dict(pid=os.getpid(),registration_sha256=lock['registration_sha256']))
    try:
        import numpy as np
        rows=json.loads((run/'inputs/source_val.json').read_text())[:256]
        hashes={r['image_path']:r['sha256'] for r in rows}
        trials=[]
        for mode in (False,True,True,False):
            decoder=(lambda rr:target_arrays(rr,hashes)) if mode else (lambda rr:np.stack([pixels(r) for r in rr]))
            digest=hashlib.sha256();began=time.monotonic();count=0
            with closing(ordered_batches(rows,decoder,parallel=mode)) as batches:
                for start,chunk,arrays in batches:
                    assert start==count and arrays.shape==(len(chunk),3,256,256)
                    count+=len(chunk);digest.update(arrays.tobytes())
            trials.append(dict(parallel=mode,seconds=time.monotonic()-began,frames=count,pixel_sha256=digest.hexdigest()))
        assert len({t['pixel_sha256'] for t in trials})==1
        cg=Path('/sys/fs/cgroup/memory')
        source=Path(p.REMOTE)/p.SOURCE_RUN
        proof=dict(passed=True,registration_sha256=lock['registration_sha256'],trials=trials,
            total_image_reads=1024,original_serial_pixels_equal_new_prefetch=True,
            speedup=sum(t['seconds'] for t in trials if not t['parallel'])/sum(t['seconds'] for t in trials if t['parallel']),
            source_state=json.loads((source/'STATE.json').read_text()),shared_training_load=True,
            nice=os.getpriority(os.PRIO_PROCESS,0),CUDA_used=False,
            memory_used=int((cg/'memory.usage_in_bytes').read_text()))
        dump(output,proof)
        print(json.dumps(proof))
    except Exception:
        dump(run/'CPU_BENCHMARK_FAILURE.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
