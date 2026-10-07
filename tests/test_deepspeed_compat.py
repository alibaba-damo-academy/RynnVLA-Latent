"""Check accumulation arithmetic using the installed DeepSpeed epilogue on CPU."""

from types import SimpleNamespace

import torch

from rynnvla.utils.deepspeed_compat import fix_zero2_gradient_accumulation
from rynnvla.utils.pipeline_parallel import ScheduleNoPipelining


def test_all_microbatches_survive_across_optimizer_steps(monkeypatch):
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    # Restore the installed method after this test's process-local patch.
    original = DeepSpeedZeroOptimizer.independent_gradient_partition_epilogue
    monkeypatch.setattr(DeepSpeedZeroOptimizer, 'independent_gradient_partition_epilogue', original)
    fix_zero2_gradient_accumulation()
    assert not fix_zero2_gradient_accumulation()
    epilogue = DeepSpeedZeroOptimizer.independent_gradient_partition_epilogue
    noop = lambda *args, **kwargs: None
    opt = SimpleNamespace(
        cpu_offload=False, overlap_comm=False, params_already_reduced=[], bit16_groups=[[]],
        params_in_partition=[[]], averaged_gradients={}, all_grad_tensors={},
        gradient_accumulation_dtype=torch.float32, first_offset=[0], partition_size=[1],
        report_ipg_memory_usage=noop, reduce_ipg_grads=noop, _release_ipg_buffers=noop, zero_grad=noop,
    )
    opt.get_all_grad_tensors = lambda *args, **kwargs: [opt.current.clone()]
    opt.get_flat_partition = lambda *args, **kwargs: [x.clone() for x in opt.all_grad_tensors[0]]

    class Engine:
        def set_gradient_accumulation_boundary(self, is_boundary):
            opt.is_gradient_accumulation_boundary = is_boundary

    class Stage:
        def forward_one_chunk(self, batches, batch_index):
            opt.current = batches[batch_index] / len(batches)
            return batches[batch_index], None

        def backward_one_chunk(self, batch_index):
            epilogue(opt)

    for gradients in ([1, 2, 3, 4], [4, -2, 8, -6]):
        batches = [torch.tensor([value], dtype=torch.float32) for value in gradients]
        ScheduleNoPipelining([Stage()], Engine()).step(batches)
        assert opt.averaged_gradients[0][0].item() == sum(gradients) / len(gradients)
        opt.averaged_gradients.clear()


def test_older_epilogue_is_left_unchanged(monkeypatch):
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    def legacy_epilogue(self):
        self.averaged_gradients[0].add_(self.current)

    monkeypatch.setattr(DeepSpeedZeroOptimizer, 'independent_gradient_partition_epilogue', legacy_epilogue)
    assert not fix_zero2_gradient_accumulation()
    assert DeepSpeedZeroOptimizer.independent_gradient_partition_epilogue is legacy_epilogue
