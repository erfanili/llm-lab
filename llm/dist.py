"""Hand-written data parallelism on top of torch.distributed collectives.

Every rank holds a full replica of the model and optimizer. All gradients live in one
contiguous fp32 buffer; each param.grad is a view into it, so autograd accumulates into
the buffer directly and a single all_reduce averages every gradient at once.
(This is the core of what torch DDP does, minus bucketing and comm/compute overlap.)
"""

import os

import torch
import torch.distributed as dist


def setup() -> tuple[int, int, torch.device]:
    """Initialise from torchrun env vars; falls back to a single process."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda", 0)
    # On this machine PCIe peer-to-peer between GPUs hangs NCCL (IOMMU/ACS); route via host memory.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


class GradBuffer:
    def __init__(self, params: list[torch.nn.Parameter]):
        self.params = params
        n = sum(p.numel() for p in params)
        self.flat = torch.zeros(n, dtype=torch.float32, device=params[0].device)
        offset = 0
        for p in params:
            assert p.dtype == torch.float32
            p.grad = self.flat[offset : offset + p.numel()].view_as(p)
            offset += p.numel()

    def zero(self):
        self.flat.zero_()

    def check_views(self):
        """Autograd must have accumulated in place; if it replaced a .grad the sync would silently skip it."""
        base = self.flat.data_ptr()
        offset = 0
        for p in self.params:
            assert p.grad.data_ptr() == base + offset * 4, "param.grad no longer views the flat buffer"
            offset += p.numel()

    def all_reduce_mean(self):
        if world_size() > 1:
            dist.all_reduce(self.flat, op=dist.ReduceOp.AVG)

    def clip_(self, max_norm: float) -> float:
        """Global-norm clipping. Grads are identical on all ranks after all_reduce, so no extra comm."""
        norm = self.flat.norm()
        self.flat.mul_(torch.clamp(max_norm / (norm + 1e-6), max=1.0))
        return norm.item()


def broadcast_params(model: torch.nn.Module):
    """Make every rank start from rank 0's weights."""
    if world_size() > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t
