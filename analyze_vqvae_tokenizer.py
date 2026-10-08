"""Compatibility entrypoint for the canonical saved-run tokenizer evaluator."""

# Keep CLI behavior and public analysis helpers in one implementation.
from scripts.analyze_vqvae_tokenizer import *  # noqa: F403
from scripts.analyze_vqvae_tokenizer import main


if __name__ == "__main__":
    main()
