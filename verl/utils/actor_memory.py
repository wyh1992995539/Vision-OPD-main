"""Opt-in actor memory diagnostics and bounded optimizer-state residency."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import time


class StageMemoryRecorder:
    """Interval peaks between synchronized markers; disabled mode never touches CUDA.

    Resetting CUDA peak counters affects process-global counters. Cumulative maxima
    are retained here and must be included in the worker's existing perf metrics.
    Device free/total is device-wide; allocated/reserved is this process only.
    """
    def __init__(self, directory=None, backend=None, rank=0):
        self.enabled = bool(directory)
        self.backend = backend
        self.path = Path(directory) / f'rank{rank}.pid{os.getpid()}.jsonl' if directory else None
        self.context = {}
        self.previous = None
        self.peak_allocated = self.peak_reserved = 0
        if self.enabled:
            if backend is None:
                import torch
                self.backend = torch.cuda
            if not self.backend.is_available():
                raise RuntimeError('Memory stage profiling requires CUDA')
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def mark(self, name, **context):
        if not self.enabled:
            return
        cuda = self.backend
        cuda.synchronize()
        allocated, reserved = cuda.memory_allocated(), cuda.memory_reserved()
        peak_a, peak_r = cuda.max_memory_allocated(), cuda.max_memory_reserved()
        self.peak_allocated = max(self.peak_allocated, peak_a)
        self.peak_reserved = max(self.peak_reserved, peak_r)
        free, total = cuda.mem_get_info()
        event = dict(time_unix=time.time(), monotonic_seconds=time.monotonic(),
                     pid=os.getpid(), event=name, interval_start=self.previous,
                     allocated_bytes=allocated, reserved_bytes=reserved,
                     interval_peak_allocated_bytes=peak_a, interval_peak_reserved_bytes=peak_r,
                     device_free_bytes=free, device_total_bytes=total,
                     synchronization_enabled=True, **self.context, **context)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        cuda.reset_peak_memory_stats()
        self.previous = name


class OptimizerResidency:
    """Keep optimizer states on CPU outside optimizer steps. Cleanup covers partial loads."""
    def __init__(self, load, offload, recorder):
        self.load, self.offload, self.recorder = load, offload, recorder
        self.active = False

    @contextmanager
    def step(self):
        if self.active:
            raise RuntimeError('Nested optimizer residency is not supported')
        self.active = True
        try:
            self.recorder.mark('optimizer_load/before')
            self.load()
            self.recorder.mark('optimizer_load/after')
            yield
        except BaseException as original:
            try:
                self.offload()
            except BaseException as cleanup:
                original.add_note(f'Optimizer offload also failed: {cleanup!r}')
            raise
        else:
            try:
                self.recorder.mark('optimizer_offload/before')
            finally:
                # Cleanup also runs when diagnostic I/O fails.
                # Callback waits for asynchronous D2H copies before returning.
                self.offload()
            self.recorder.mark('optimizer_offload/after')
        finally:
            self.active = False


def profiled_stage(name):
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            recorder = getattr(self, 'memory_recorder', None)
            if recorder is None or not recorder.enabled:
                return fn(self, *args, **kwargs)
            label = name
            if name == 'forward':
                label = 'teacher_forward' if kwargs.get('module') is not None else 'student_forward'
                batch = args[0] if args else kwargs['micro_batch']
                recorder.context.update(
                    micro_batch_samples=int(batch['input_ids'].shape[0]),
                    sequence_width=int(batch['input_ids'].shape[-1]),
                    max_unpadded_sequence_tokens=int(batch['attention_mask'].sum(-1).max().item()),
                    response_width=int(batch['responses'].shape[-1]),
                )
            recorder.mark(label + '/before')
            try:
                result = fn(self, *args, **kwargs)
            except BaseException as original:
                try:
                    recorder.mark(label + '/error')
                except BaseException as diagnostic:
                    original.add_note(f'Memory diagnostic also failed: {diagnostic!r}')
                raise
            recorder.mark(label + '/after')
            return result
        return call
    return decorate


def with_optimizer_residency(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        manager = getattr(self, 'optimizer_residency', None)
        with manager.step() if manager is not None else nullcontext():
            return fn(self, *args, **kwargs)
    return call
