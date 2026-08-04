from collections.abc import Iterable

import numpy as np
import torch as T
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from torch.utils.data import default_collate

from heptokens.models.vq_vae import LitVqVae


def collate_and_transform(
    batch: Iterable[dict],
    do_default_collate: bool = True,
    transforms: dict | DictConfig | list | None = None,
) -> dict:
    from omegaconf import ListConfig

    if do_default_collate:
        batch = default_collate(batch)
    if transforms is not None:
        # Handle OmegaConf DictConfig, dict, and list of transforms
        if isinstance(transforms, (dict, DictConfig)):
            # Extract callable transforms from dict/DictConfig, filtering out non-callable values
            transform_list = [v for v in transforms.values() if callable(v)]
        elif isinstance(transforms, (list, ListConfig)):
            transform_list = list(transforms)
        else:
            transform_list = [transforms]

        for transform in transform_list:
            if not callable(transform):
                raise TypeError(
                    f"collate_and_transform: expected a callable transform, got "
                    f"{type(transform).__name__!r}: {transform!r}.  "
                    f"Ensure all transforms are fully instantiated before being "
                    f"passed to the dataloader."
                )
            batch = transform(batch)
    return batch


class VqvaeTokenizer:
    """Add VQ-VAE token IDs to the batch using a trained checkpoint."""

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path
        self._model: LitVqVae | None = None

    def _get_model(self) -> LitVqVae:
        if self._model is None:
            device = T.device("cuda" if T.cuda.is_available() else "cpu")
            self._model = LitVqVae.load_from_checkpoint(self.ckpt_path, map_location=device)
            self._model.to(device)
            self._model.eval()
        return self._model

    def __call__(self, jet_dict: dict[T.Tensor]) -> dict:
        model = self._get_model()
        device = next(model.parameters()).device
        with T.no_grad():
            batch = {
                k: (v.to(device) if isinstance(v, T.Tensor) else v) for k, v in jet_dict.items()
            }
            _, indices, _ = model.encode(batch)
            # indices = indices.to("cpu")  # NOTE: this is needed if running in preprocessing.
        jet_dict["tokens"] = indices
        return jet_dict


def preprocess_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
    jet_fn: BaseEstimator,
) -> dict:
    """Preprocess a batch of jets already stored as pytorch tensors."""
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    jets = jet_dict["jets"]

    # Convert to numpy for sklearn
    csts_np = csts.cpu().numpy() if isinstance(csts, T.Tensor) else csts
    jets_np = jets.cpu().numpy() if isinstance(jets, T.Tensor) else jets

    # Pad and transform
    if (feat_diff := cst_fn.n_features_in_ - csts_np.shape[-1]) > 0:
        zeros = np.zeros((csts_np.shape[:-1] + (feat_diff,)), dtype=csts_np.dtype)
        csts_np = np.concatenate((csts_np, zeros), axis=-1)

    csts_np[mask.cpu().numpy() if isinstance(mask, T.Tensor) else mask] = cst_fn.transform(
        csts_np[mask.cpu().numpy() if isinstance(mask, T.Tensor) else mask]
    )

    if feat_diff > 0:
        csts_np = csts_np[:, :-feat_diff]

    # Convert back to tensor
    jet_dict["csts"] = T.from_numpy(csts_np).float()
    jet_dict["jets"] = T.from_numpy(jet_fn.transform(jets_np)).float()

    return jet_dict


def preprocess_objects_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
) -> dict:
    """Preprocess object-tokenizer features.

    Atlas object-tokenizer batches store the per-object inputs in ``csts`` and
    validity in ``mask``. Event inputs live in ``jets`` but are not used by the
    VQ-VAE tokenizer, so this transform intentionally touches only ``csts``.
    """
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    csts_np = csts.detach().cpu().numpy().copy() if isinstance(csts, T.Tensor) else csts.copy()
    mask_np = mask.cpu().numpy() if isinstance(mask, T.Tensor) else mask

    if mask_np.any():
        csts_np[mask_np] = cst_fn.transform(csts_np[mask_np])

    jet_dict = jet_dict.copy()
    jet_dict["csts"] = T.from_numpy(csts_np).float()
    return jet_dict


def inverse_preprocess_objects_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
) -> dict:
    """Invert object-tokenizer preprocessing on valid ``csts`` entries."""
    csts = jet_dict["csts"].clone()
    mask = jet_dict["mask"]

    if mask.any():
        inverse_csts = cst_fn.inverse_transform(csts[mask].cpu().numpy())
        csts[mask] = T.from_numpy(inverse_csts).float()

    jet_dict_inverse = jet_dict.copy()
    jet_dict_inverse["csts"] = csts
    return jet_dict_inverse


def inverse_preprocess_batch(
    jet_dict: dict[T.Tensor],
    cst_fn: BaseEstimator,
    jet_fn: BaseEstimator,
) -> dict:
    """Apply inverse preprocessing to a batch of jets.

    Args:
        jet_dict: Dictionary containing 'csts', 'jets', and 'mask' tensors
        cst_fn: Fitted QuantileTransformer for constituents
        jet_fn: Fitted QuantileTransformer for jets

    Returns:
        Dictionary with inverse-transformed constituents and jets
    """
    csts = jet_dict["csts"].clone()  # Clone to avoid modifying original
    mask = jet_dict["mask"]
    jets = jet_dict["jets"].clone()

    # Inverse transform constituents
    # Only transform valid (masked) constituents
    if mask.any():
        valid_csts = csts[mask].cpu().numpy()
        inverse_csts = cst_fn.inverse_transform(valid_csts)
        csts[mask] = T.from_numpy(inverse_csts).float()

    # Inverse transform jets
    inverse_jets = jet_fn.inverse_transform(jets.cpu().numpy())
    jets = T.from_numpy(inverse_jets).float()

    # Update dictionary
    jet_dict_inverse = jet_dict.copy()
    jet_dict_inverse["csts"] = csts
    jet_dict_inverse["jets"] = jets

    return jet_dict_inverse


def mask_batch(
    jet_dict: dict,
    mask_fraction: float = 0.4,
    key: str = "null_mask",
) -> dict:
    """Applies a masking function of a batch of jets.

    Will add a new key to the jet_dict with the locations of the new mask.
    """
    null_mask = T.stack([mask_jet(mask, mask_fraction=mask_fraction) for mask in jet_dict["mask"]])
    jet_dict[key] = null_mask
    return jet_dict


def mask_jet(
    mask: T.Tensor,
    mask_fraction: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    if seed is not None:
        T.manual_seed(seed)
    max_drop = int(T.floor(mask_fraction * mask.sum()))
    null_mask = T.zeros_like(mask, dtype=T.bool)

    # Exit now if we are not dropping any nodes
    if max_drop == 0:
        return null_mask

    # Generate a random score per node, the lowest frac will be killed
    rand = T.rand(len(mask))
    rand[~mask] = 9999
    drop_idx = T.argsort(rand)[:max_drop]
    null_mask[drop_idx] = True

    return null_mask
