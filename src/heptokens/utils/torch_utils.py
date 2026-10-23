"""Mix of utility functions specifically for pytorch."""

import contextlib
import math
import os
import re
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch as T
import torch.optim.lr_scheduler as schd
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, Subset, random_split
from torch.utils.data.dataloader import default_collate

from heptokens.utils.schedulers import CyclicWithWarmup, LinearWarmupRootDecay, WarmupToConstant


class ParameterNoWD(nn.Parameter):
    """A parameter which does not have weight decay.

    Must be used alongside the mltools.optimisers.AdamWS optimizer for this to take
    effect. As this is a subclass of nn.Parameter, it can be used in the same way.
    """


def rms_norm(x, dim: int | tuple = -1, eps: float = 1e-6) -> T.Tensor:
    """Normalise the vector to have unit variance."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n, alpha=math.sqrt(n.numel() / x.numel()))
    return x / n.to(x.dtype)


def rms(x, dim: int | tuple = -1, eps: float = 1e-6) -> T.Tensor:
    """Calculate the RMS of the vector."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    return T.add(eps, n, alpha=math.sqrt(n.numel() / x.numel()))


def get_activations(
    model: nn.Module,
    activation_dict: dict,
    regex: list | None = None,
    types: list | None = None,
) -> list:
    """Create hooks for storing output activation magnitudes of layers in a model."""
    hooks = []

    def hook(name) -> callable:
        def forward_hook(_module: nn.Module, _input: T.Tensor, output: T.Tensor):
            norm = rms(output.detach()).mean().cpu().item()
            activation_dict[name] = norm

        return forward_hook

    for n, m in model.named_modules():
        passed = False
        if regex is not None and any(re.match(r, n) for r in regex):
            passed = True
        if types is not None and any(isinstance(m, t) for t in types):
            passed = True
        if passed:
            h = m.register_forward_hook(hook(n))
            hooks.append(h)

    return hooks


def remove_hooks(hooks: list) -> None:
    """Remove a list of hooks."""
    for hook in hooks:
        hook.remove()


def gradient_norm(model) -> float:
    """Return the total norm of the gradients of a model.

    The strange logic is to avoid upscaling the norm when using mixed precision.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_data = p.grad.detach().data.square().sum()
            if total_norm == 0:
                total_norm = grad_data
            else:
                total_norm += grad_data
    if total_norm == 0:
        return 0
    return total_norm.sqrt().item()


def get_submodules(module: nn.Module, depth: int = 1, prefix="") -> list:
    """Return a list of all of the base modules in a network."""
    modules = []
    if depth == 0 or not list(module.children()):
        return [(prefix, module)]
    for n, child in module.named_children():
        subname = prefix + ("." if prefix else "") + n
        modules.extend(get_submodules(child, depth - 1, subname))
    return modules


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.data.zero_()
    return module


@contextmanager
def set_eval(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def reset_params(layer: nn.Module) -> None:
    """Reset the parameters of a pytorch layer."""
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()


def occupancy(probs: T.Tensor, dim: int = -1):
    """Return the number of unique argmax values given a probability tensor."""
    num_dims = probs.shape[dim]
    return T.unique(T.argmax(probs, dim=-1)).size(0) / num_dims


def append_dims(x: T.Tensor, target_dims: int, dim=-1) -> T.Tensor:
    """Append dimensions of size 1 to tensor until it has target_dims."""
    if (dim_diff := target_dims - x.dim()) < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")

    # Fast exit conditions
    if dim_diff == 0:
        return x
    if dim_diff == 1:
        return x.unsqueeze(dim)
    if dim == -1:
        return x[(...,) + (None,) * dim_diff]
    if dim == 0:
        return x[(None,) * dim_diff + (...)]

    # Check if the dimension is in range
    allow = [-x.dim() - 1, x.dim()]
    if not allow[0] <= dim <= allow[1]:
        raise IndexError(f"Dimension out of range (expected to be in {allow} but got {dim})")

    # Following only works for a positive index
    if dim < 0:
        dim += x.dim() + 1
    return x.view(*x.shape[:dim], *dim_diff * (1,), *x.shape[dim:])


def attach_context(x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
    """Concat a tensor with context which has the same or lower dimensions.

    New dimensions are added at index 1
    """
    if ctxt is None:
        return x
    ctxt = append_dims(ctxt, x.dim(), dim=1)
    ctxt = ctxt.expand(*x.shape[:-1], -1)
    return T.cat((x, ctxt), dim=-1)


def dtype_lookup(dtype: Any) -> T.dtype:
    """Return a torch dtype based on a string."""
    return {
        "double": T.float64,
        "float": T.float32,
        "half": T.float16,
        "int": T.int32,
        "long": T.int64,
    }[dtype]


class GradsOff:
    """Context manager for passing through a model without it tracking gradients."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self) -> None:
        self.model.requires_grad_(False)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.model.requires_grad_(True)


def base_modules(module: nn.Module) -> list:
    """Return a list of all of the base modules in a network.

    Different to both children() and modules().
    Children() returns all of the direct children of a module.
    Modules() returns all of the modules in a network, including all intermediates.
    """
    total = []
    children = list(module.children())
    if not children:
        total += [module]
    else:
        for c in children:
            total += base_modules(c)
    return total


def empty_0dim_like(inpt: T.Tensor | np.ndarray) -> T.Tensor | np.ndarray:
    """Return an empty tensor with same size as input but with final dim = 0."""
    # Get all but the final dimension
    all_but_last = inpt.shape[:-1]

    # Ensure that this is a tuple/list so it can agree with return syntax
    if isinstance(all_but_last, int):
        all_but_last = [all_but_last]

    if isinstance(inpt, T.Tensor):
        return T.empty((*all_but_last, 0), dtype=inpt.dtype, device=inpt.device)
    return np.empty((*all_but_last, 0))


def get_sched(
    sched_dict: dict,
    opt: Optimizer,
    steps_per_epoch: int = 0,
    max_lr: float | None = None,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> schd._LRScheduler:
    """Return a pytorch learning rate schedular given a dict containing a name and other
    kwargs.

    I still prefer this method as opposed to the hydra implementation as
    it allows us to specify the cyclical scheduler periods as a function of epochs
    rather than steps.

    Parameters
    ----------
    sched_dict : dict
        A dictionary of kwargs used to select and configure the scheduler.
    opt : Optimizer
        The optimizer to apply the learning rate to.
    steps_per_epoch : int
        The number of minibatches in a single training epoch.
    max_lr : float, optional
        The maximum learning rate for the one shot. Only for OneCycle learning.
    max_epochs : int, optional
        The maximum number of epochs to train for. Only for OneCycle learning.
    max_steps : int, optional
        The maximum number of steps to train for. Only for OneCycle learning.
    """
    # Pop off the name and learning rate for the optimizer
    dict_copy = sched_dict.copy()
    name = dict_copy.pop("name")

    # Get the max_lr from the optimizer if not specified
    max_lr = max_lr or opt.defaults["lr"]

    # Exit if the name indicates no scheduler
    if name in {"", "none", "None"}:
        return None

    # If the steps per epoch is 0, try and get it from the sched_dict
    if steps_per_epoch == 0:
        try:
            steps_per_epoch = dict_copy.pop("steps_per_epoch")
        except KeyError as e:
            raise ValueError(
                "steps_per_epoch was not passed to get_sched and was ",
                "not in the scheduler dictionary!",
            ) from e

    # If max_steps is not specified, then use the max_epochs
    max_steps = max_steps if max_steps > 1 else steps_per_epoch * max_epochs

    # Pop off the number of epochs per cycle (needed as arg)
    if "epochs_per_cycle" in dict_copy:
        epochs_per_cycle = dict_copy.pop("epochs_per_cycle")
    else:
        epochs_per_cycle = 1

    # Use the same div_factor for cyclic with warmup
    if name == "cyclicwithwarmup" and "div_factor" not in dict_copy:
        dict_copy["div_factor"] = 1e4

    if name == "cosann":
        return schd.CosineAnnealingLR(opt, steps_per_epoch * epochs_per_cycle, **dict_copy)
    if name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    if name == "onecycle":
        return schd.OneCycleLR(opt, max_lr, total_steps=max_steps, **dict_copy)
    if name == "cyclicwithwarmup":
        return CyclicWithWarmup(
            opt, max_lr, total_steps=steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    if name == "linearwarmuprootdecay":
        return LinearWarmupRootDecay(opt, **dict_copy)
    if name == "warmup":
        return WarmupToConstant(opt, **dict_copy)
    if name == "lr_scheduler.ExponentialLR":
        return schd.ExponentialLR(opt, **dict_copy)
    if name == "lr_scheduler.ConstantLR":
        return schd.ConstantLR(opt, **dict_copy)
    raise ValueError(f"No scheduler with name: {name}")


def train_valid_split(
    dataset: Dataset, v_frac: float, split_type="interweave"
) -> tuple[Subset, Subset]:
    """Split a pytorch dataset into a training and validation pytorch Subsets.

    Parameters
    ----------
    dataset:
        The dataset to split
    v_frac:
        The validation fraction, reciprocals of whole numbers are best
    split_type: The type of splitting for the dataset. Default is interweave.
        basic: Take the first x event for the validation
        interweave: The every x events for the validation
        rand: Use a random splitting method (seed 42)
    """
    if split_type == "rand":
        v_size = int(v_frac * len(dataset))
        t_size = len(dataset) - v_size
        return random_split(dataset, [t_size, v_size], generator=T.Generator().manual_seed(42))
    if split_type == "basic":
        v_size = int(v_frac * len(dataset))
        valid_indxs = np.arange(0, v_size)
        train_indxs = np.arange(v_size, len(dataset))
        return Subset(dataset, train_indxs), Subset(dataset, valid_indxs)
    if split_type == "interweave":
        v_every = int(1 / v_frac)
        valid_indxs = np.arange(0, len(dataset), v_every)
        train_indxs = np.delete(np.arange(len(dataset)), np.s_[::v_every])
        return Subset(dataset, train_indxs), Subset(dataset, valid_indxs)
    raise ValueError(f"Split type {split_type} not recognised!")


def k_fold_split(dataset: Dataset, num_folds: int, fold_idx: int) -> tuple[Subset, Subset, Subset]:
    """Perform a k-fold cross."""
    assert num_folds > 2
    assert fold_idx < num_folds

    test_fold = fold_idx
    val_fold = (fold_idx + 1) % num_folds
    train_folds = [i for i in range(num_folds) if i not in {fold_idx, val_fold}]

    data_idxes = np.arange(len(dataset))
    in_k = data_idxes % num_folds

    test = Subset(dataset, data_idxes[in_k == test_fold])
    valid = Subset(dataset, data_idxes[in_k == val_fold])
    train = Subset(dataset, data_idxes[np.isin(in_k, train_folds)])

    return train, valid, test


def to_device(
    x: T.Tensor | tuple | list | dict | None,
    device: str | T.device,
) -> T.Tensor | tuple | list | dict:
    """Return a copy of a tensor on the chosen device.

    - Calls pytorch's .to() but allows for values to be nested lists, tuples or dicts.
    """
    if x is None:
        return x
    if isinstance(x, T.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (tuple | list)):
        return type(x)(to_device(v) for v in x)
    raise TypeError(f"Unsupported input type: {type(x)}")


def to_numpy(
    x: T.Tensor | np.ndarray | tuple | list | dict,
) -> np.ndarray | dict | list | None:
    """Convert a tensor to a numpy array.

    - Includes gradient deletion, and device migration and supportes nested structures.
    """
    if x is None or isinstance(x, np.ndarray):
        return x
    if isinstance(x, T.Tensor):
        if x.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
            x = x.half()
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (tuple | list)):
        return type(x)(to_numpy(v) for v in x)
    raise TypeError(f"Unsupported input type: {type(x)}")


def print_gpu_info(dev=0):
    """Print the current gpu usage."""
    total = T.cuda.get_device_properties(dev).total_memory / 1024**3
    reser = T.cuda.memory_reserved(dev) / 1024**3
    alloc = T.cuda.memory_allocated(dev) / 1024**3
    print(f"\nTotal = {total:.2f}\nReser = {reser:.2f}\nAlloc = {alloc:.2f}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a pytorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module, norm_type: float = 2.0):
    """Return the norm of the gradients of a given model."""
    return to_numpy(
        T.norm(
            T.stack([T.norm(p.grad.detach(), norm_type) for p in model.parameters()]),
            norm_type,
        )
    )


def reparam_trick(tensor: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Apply the reparameterisation trick to split a tensor into means and devs.

    - Returns a sample, the means and devs as a tuple
    - Splitting is along the final dimension
    - Used primarily in variational autoencoders
    """
    means, lstds = T.chunk(tensor, 2, dim=-1)
    latents = means + T.randn_like(means) * lstds.exp()
    return latents, means, lstds


def get_max_cpu_suggest():
    """Try to compute a suggested max number of worker based on system's resources."""
    suggest = None
    with contextlib.suppress((OSError, AttributeError, ValueError)):
        suggest = len(os.sched_getaffinity(0))
    if suggest is None:
        suggest = os.cpu_count()
    if suggest is not None:
        suggest -= 1
    return suggest


def log_squash(data: T.Tensor) -> T.Tensor:
    """Apply a log squashing function for distributions with high tails."""
    return T.sign(data) * T.log(T.abs(data) + 1)


def torch_undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return T.sign(data) * (T.exp(T.abs(data)) - 1)


@T.no_grad()
def ema_param_sync(source: nn.Module, target: nn.Module, ema_decay: float) -> None:
    """Synchronize the parameters of two modules using exponential moving average (EMA).

    Parameters
    ----------
    source : nn.Module
        The source module whose parameters are used to update the target module.
    target : nn.Module
        The target module whose parameters are updated.
    ema_decay : float
        The decay rate for the EMA update.
    """
    for s_params, t_params in zip(source.parameters(), target.parameters(), strict=False):
        t_params.data.copy_(t_params.data.lerp_(s_params.data, 1 - ema_decay))


def masked_mean(x: T.Tensor, mask: T.BoolTensor, dim: int = -1) -> T.Tensor:
    """Return the mean of a tensor along a dimension, ignoring masked elements."""
    mask_dim = dim + 1 * (dim < 0)
    total = (x * mask.unsqueeze(-1)).sum(dim)
    return total / mask.sum(mask_dim, keepdim=True)


def preprocess_and_collate(batch: Iterable, preprocessing: Callable | None = None) -> Any:
    """Apply collation with an extra preprocessing fn to each sample."""
    if preprocessing is not None:
        batch = [preprocessing(x) for x in batch]
    return default_collate(batch)


def squash_fn(x: T.Tensor | np.ndarray, a: float) -> np.ndarray:
    """Hand crafted squash function for bounded data in the range (0, 1)."""
    if a == 1:
        return x
    assert a >= 1
    x[x < 0] = 0  # Works on both tensors and numpy arrays
    x[x > 1] = 1
    return (1 - (1 - x) ** a) ** (1 / a)


def unsquash_fn(x: T.Tensor | np.ndarray, a: float) -> np.ndarray:
    """Undo the squash function above."""
    if a == 1:
        return x
    assert a >= 1
    x[x < 0] = 0  # Works on both tensors and numpy arrays
    x[x > 1] = 1
    return 1 - (1 - x**a) ** (1 / a)


def dict_to_device(data_dict, device):
    """Move all tensors in a dict to the specified device."""
    return {k: v.to(device) if T.is_tensor(v) else v for k, v in data_dict.items()}
