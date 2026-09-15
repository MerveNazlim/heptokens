"""Basic training script."""

import inspect
import logging
import warnings

import hydra
import lightning.pytorch as pl
import torch as T
from omegaconf import DictConfig, OmegaConf

from heptokens.utils.hydra import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
    save_declaration,
)

log = logging.getLogger(__name__)
# Suppress torchvision image library warnings (we don't use image functionality)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


def feature_names_from_datamodule(cfg: DictConfig) -> list[str] | None:
    """Best-effort feature names for tokenizer diagnostics."""
    datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
    collections = datamodule.get("object_collections") or []
    object_type = datamodule.get("object_type")
    if datamodule.get("output_mode") != "object" or object_type is None:
        return None
    for collection in collections:
        if collection.get("object_name") == object_type:
            return [str(path).split("/")[-1] for path in collection.get("inputs") or []]
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main training script."""
    log.info("Setting up full job config")

    if cfg.full_resume:
        log.info("Attempting to resume previous job")
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    log.info(f"Setting seed to: {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    if cfg.weight_ckpt_path:
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        model_class = hydra.utils.get_class(cfg.model._target_)
        model = model_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    else:
        model_class = hydra.utils.get_class(cfg.model._target_)
        model_kwargs = {
            "data_sample": datamodule.get_data_sample(),
            "n_classes": datamodule.get_n_classes(),
        }
        if "feature_names" in inspect.signature(model_class.__init__).parameters:
            model_kwargs["feature_names"] = feature_names_from_datamodule(cfg)
        if "token_vocabulary" in inspect.signature(model_class.__init__).parameters:
            get_vocabulary = getattr(datamodule, "get_token_vocabulary", None)
            if get_vocabulary is not None:
                token_vocabulary = get_vocabulary()
                model_kwargs["token_vocabulary"] = token_vocabulary
                if token_vocabulary is not None and "vocab_size" in inspect.signature(
                    model_class.__init__
                ).parameters:
                    model_kwargs["vocab_size"] = int(token_vocabulary["vocab_size"])
        model = hydra.utils.instantiate(cfg.model, **model_kwargs)

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging all hyperparameters")
    log_hyperparameters(cfg, model, trainer)
    log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Checking if training finished correctly")
    if trainer.state.status == "finished":
        log.info(" -- YES!! -- ")
        save_declaration()


if __name__ == "__main__":
    main()
