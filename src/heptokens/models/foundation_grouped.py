"""Masked pretraining with one sequence position per physics object."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.transformer import Transformer
from heptokens.models.utils import ScheduledOptimiserMixin


class GroupedSequenceBackbone(nn.Module):
    """Concatenate residual-quantizer embeddings before event attention."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_quantizers: int,
        quantizer_embedding_dim: int | None,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_length: int,
        num_type_ids: int,
        pad_token_id: int,
        use_type_embedding: bool,
        use_position_embedding: bool,
    ) -> None:
        super().__init__()
        self.max_quantizers = max_quantizers
        self.pad_token_id = pad_token_id
        embedding_dim = quantizer_embedding_dim or math.ceil(hidden_dim / max_quantizers)
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
        self.object_projection = nn.Linear(max_quantizers * embedding_dim, hidden_dim)
        self.type_embedding = (
            nn.Embedding(num_type_ids, hidden_dim) if use_type_embedding else None
        )
        self.position_embedding = (
            nn.Embedding(max_seq_length, hidden_dim) if use_position_embedding else None
        )
        self.transformer = Transformer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            d_model=hidden_dim,
            n_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected grouped tokens [B, N, Q], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.max_quantizers:
            raise ValueError(
                f"Expected Q={self.max_quantizers}, got grouped tokens with Q={tokens.shape[-1]}"
            )

        batch_size, seq_len, _ = tokens.shape
        quantizer_mask = tokens.ne(self.pad_token_id)
        embedded = self.token_embedding(tokens)
        embedded = embedded * quantizer_mask.unsqueeze(-1)
        x = self.object_projection(embedded.flatten(start_dim=2))

        if self.type_embedding is not None:
            if type_ids is None:
                type_ids = torch.zeros_like(mask, dtype=torch.long)
            x = x + self.type_embedding(type_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.position_embedding(positions)

        x = self.dropout(x)
        return self.layer_norm(self.transformer(x, mask=mask.bool()))


class GroupedAutoregressiveQuantizerDecoder(nn.Module):
    """Predict ordered RVQ codes with teacher forcing and legal local heads."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_embedding_dim: int,
        max_quantizers: int,
        pad_token_id: int,
        token_vocabulary: dict,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_quantizers = max_quantizers
        self.pad_token_id = pad_token_id
        self.object_specs: dict[tuple[int, int], tuple[int, int]] = {}
        self.object_heads = nn.ModuleDict()

        for object_spec in token_vocabulary.get("objects", {}).values():
            type_id = int(object_spec["type_id"])
            for quantizer in object_spec.get("quantizers", []):
                index = int(quantizer["index"])
                if index >= max_quantizers:
                    raise ValueError(
                        f"Vocabulary quantizer q{index} exceeds max_quantizers="
                        f"{max_quantizers}"
                    )
                base = int(quantizer["base"])
                size = int(quantizer["size"])
                key = self.object_key(type_id, index)
                self.object_specs[(type_id, index)] = (base, size)
                self.object_heads[key] = nn.Linear(hidden_dim, size)

        if not self.object_specs:
            raise ValueError(
                "Autoregressive quantizer decoding requires object codebook ranges in "
                "token_vocabulary"
            )

        self.event_specs: dict[int, tuple[int, int, int]] = {}
        self.event_heads = nn.ModuleDict()
        event_offset = 1 if token_vocabulary.get("includes_cls", False) else 0
        event_type_id = int(token_vocabulary.get("event", {}).get("type_id", 1))
        event_tokens = token_vocabulary.get("event_tokens") or token_vocabulary.get(
            "event", {}
        ).get("tokens", [])
        for event_index, event_spec in enumerate(event_tokens):
            position = event_offset + event_index
            base = int(event_spec["base"])
            size = int(event_spec["size"])
            self.event_specs[position] = (event_type_id, base, size)
            self.event_heads[self.event_key(position)] = nn.Linear(hidden_dim, size)

        self.quantizer_prediction_embedding = nn.Embedding(max_quantizers, hidden_dim)
        self.previous_projection = nn.Linear(token_embedding_dim, hidden_dim)
        self.autoregressive_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    @staticmethod
    def object_key(type_id: int, quantizer: int) -> str:
        return f"type{type_id}_q{quantizer}"

    @staticmethod
    def event_key(position: int) -> str:
        return f"position{position}"

    @property
    def object_type_ids(self) -> set[int]:
        return {type_id for type_id, _ in self.object_specs}

    def maskable_positions(self, type_ids: torch.Tensor) -> torch.Tensor:
        """Return positions supported by an object or event-context output head."""
        supported = torch.zeros_like(type_ids, dtype=torch.bool)
        for type_id in self.object_type_ids:
            supported |= type_ids.eq(type_id)
        if self.event_specs:
            positions = torch.arange(type_ids.shape[1], device=type_ids.device)
            for position, (type_id, _, _) in self.event_specs.items():
                supported |= positions.unsqueeze(0).eq(position) & type_ids.eq(type_id)
        return supported

    def _teacher_forced_states(
        self,
        context: torch.Tensor,
        labels: torch.Tensor,
        token_embedding: nn.Embedding,
    ) -> torch.Tensor:
        previous = torch.full_like(labels, self.pad_token_id)
        previous[:, 1:] = labels[:, :-1]
        previous_embedded = token_embedding(previous)
        decoder_input = self.previous_projection(previous_embedded)
        quantizer_ids = torch.arange(self.max_quantizers, device=context.device)
        quantizer_embedding = self.quantizer_prediction_embedding(quantizer_ids)
        decoder_input = decoder_input + quantizer_embedding.unsqueeze(0)
        states, _ = self.autoregressive_gru(decoder_input, context.unsqueeze(0))
        return states + quantizer_embedding.unsqueeze(0)

    def teacher_forced_metrics(
        self,
        *,
        context: torch.Tensor,
        labels: torch.Tensor,
        valid_labels: torch.Tensor,
        type_ids: torch.Tensor,
        positions: torch.Tensor,
        token_embedding: nn.Embedding,
    ) -> dict:
        """Compute code-weighted loss and accuracy using true earlier RVQ codes."""
        states = self._teacher_forced_states(context, labels, token_embedding)
        loss_sum = context.sum() * 0.0
        correct = torch.zeros((), dtype=torch.long, device=context.device)
        count = torch.zeros((), dtype=torch.long, device=context.device)
        quantizer_loss_sums = [context.sum() * 0.0 for _ in range(self.max_quantizers)]
        quantizer_correct = [
            torch.zeros((), dtype=torch.long, device=context.device)
            for _ in range(self.max_quantizers)
        ]
        quantizer_counts = [
            torch.zeros((), dtype=torch.long, device=context.device)
            for _ in range(self.max_quantizers)
        ]

        for (type_id, quantizer), (base, _) in self.object_specs.items():
            selected = valid_labels[:, quantizer] & type_ids.eq(type_id)
            targets = labels[selected, quantizer] - base
            logits = self.object_heads[
                self.object_key(type_id, quantizer)
            ](states[selected, quantizer])
            selected_loss = F.cross_entropy(logits, targets, reduction="sum")
            selected_correct = logits.argmax(dim=-1).eq(targets).sum()
            selected_count = selected.sum()
            loss_sum = loss_sum + selected_loss
            correct = correct + selected_correct
            count = count + selected_count
            quantizer_loss_sums[quantizer] = (
                quantizer_loss_sums[quantizer] + selected_loss
            )
            quantizer_correct[quantizer] = (
                quantizer_correct[quantizer] + selected_correct
            )
            quantizer_counts[quantizer] = (
                quantizer_counts[quantizer] + selected_count
            )

        for position, (type_id, base, _) in self.event_specs.items():
            selected = (
                positions.eq(position)
                & type_ids.eq(type_id)
                & valid_labels[:, 0]
            )
            targets = labels[selected, 0] - base
            logits = self.event_heads[self.event_key(position)](context[selected])
            selected_loss = F.cross_entropy(logits, targets, reduction="sum")
            selected_correct = logits.argmax(dim=-1).eq(targets).sum()
            selected_count = selected.sum()
            loss_sum = loss_sum + selected_loss
            correct = correct + selected_correct
            count = count + selected_count
            quantizer_loss_sums[0] = quantizer_loss_sums[0] + selected_loss
            quantizer_correct[0] = quantizer_correct[0] + selected_correct
            quantizer_counts[0] = quantizer_counts[0] + selected_count

        denominator = count.clamp_min(1)
        return {
            "loss": loss_sum / denominator,
            "accuracy": correct.float() / denominator,
            "quantizer_loss_sums": quantizer_loss_sums,
            "quantizer_correct": quantizer_correct,
            "quantizer_counts": quantizer_counts,
        }

    def generate_object_codes(
        self,
        *,
        context: torch.Tensor,
        type_ids: torch.Tensor,
        token_embedding: nn.Embedding,
        sample: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate legal object codes sequentially from earlier predictions."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        codes = torch.full(
            (context.shape[0], self.max_quantizers),
            self.pad_token_id,
            dtype=torch.long,
            device=context.device,
        )
        previous = torch.full(
            (context.shape[0],),
            self.pad_token_id,
            dtype=torch.long,
            device=context.device,
        )
        hidden = context.unsqueeze(0)
        for quantizer in range(self.max_quantizers):
            decoder_input = self.previous_projection(token_embedding(previous))
            quantizer_embedding = self.quantizer_prediction_embedding.weight[quantizer]
            decoder_input = decoder_input + quantizer_embedding
            output, hidden = self.autoregressive_gru(
                decoder_input.unsqueeze(1), hidden
            )
            state = output[:, 0] + quantizer_embedding
            for type_id in torch.unique(type_ids).tolist():
                specification = self.object_specs.get((int(type_id), quantizer))
                if specification is None:
                    continue
                selected = type_ids.eq(type_id)
                base, _ = specification
                logits = self.object_heads[
                    self.object_key(int(type_id), quantizer)
                ](state[selected])
                if sample:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    local_codes = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    ).squeeze(1)
                else:
                    local_codes = logits.argmax(dim=-1)
                codes[selected, quantizer] = local_codes + base
            previous = codes[:, quantizer]
        return codes


class GroupedMaskedSequenceModel(nn.Module):
    """Mask one sequence position and reconstruct its grouped token codes."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_quantizers: int,
        quantizer_embedding_dim: int | None,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_length: int,
        num_type_ids: int,
        pad_token_id: int,
        use_type_embedding: bool,
        use_position_embedding: bool,
        quantizer_decoder: str = "parallel",
        token_vocabulary: dict | None = None,
    ) -> None:
        super().__init__()
        if quantizer_decoder not in {"parallel", "autoregressive"}:
            raise ValueError(
                "quantizer_decoder must be 'parallel' or 'autoregressive', got "
                f"{quantizer_decoder!r}"
            )
        self.vocab_size = vocab_size
        self.max_quantizers = max_quantizers
        self.pad_token_id = pad_token_id
        self.quantizer_decoder = quantizer_decoder
        self.backbone = GroupedSequenceBackbone(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_quantizers=max_quantizers,
            quantizer_embedding_dim=quantizer_embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_type_ids=num_type_ids,
            pad_token_id=pad_token_id,
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
        )
        if quantizer_decoder == "parallel":
            self.quantizer_prediction_embedding = nn.Embedding(
                max_quantizers, hidden_dim
            )
            self.mlm_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, vocab_size),
            )
            self.autoregressive_decoder = None
        else:
            if token_vocabulary is None:
                raise ValueError(
                    "token_vocabulary is required when quantizer_decoder="
                    "'autoregressive'"
                )
            vocabulary_size = int(token_vocabulary.get("vocab_size", vocab_size))
            if vocabulary_size != vocab_size:
                raise ValueError(
                    f"Configured vocab_size={vocab_size} does not match token vocabulary "
                    f"size {vocabulary_size}"
                )
            self.quantizer_prediction_embedding = None
            self.mlm_head = None
            self.autoregressive_decoder = GroupedAutoregressiveQuantizerDecoder(
                hidden_dim=hidden_dim,
                token_embedding_dim=self.backbone.token_embedding.embedding_dim,
                max_quantizers=max_quantizers,
                pad_token_id=pad_token_id,
                token_vocabulary=token_vocabulary,
            )

    def masked_logits(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.quantizer_decoder != "parallel":
            raise RuntimeError(
                "masked_logits is the parallel-decoder API. Use "
                "autoregressive_metrics or generate_masked_object_codes for an "
                "autoregressive model."
            )
        hidden = self.backbone(tokens, mask, type_ids)
        selected = hidden[masked_positions]
        quantizer_ids = torch.arange(self.max_quantizers, device=tokens.device)
        states = selected.unsqueeze(1) + self.quantizer_prediction_embedding(quantizer_ids)
        return self.mlm_head(states)

    def maskable_positions(self, type_ids: torch.Tensor) -> torch.Tensor:
        if self.autoregressive_decoder is None:
            return torch.ones_like(type_ids, dtype=torch.bool)
        return self.autoregressive_decoder.maskable_positions(type_ids)

    def autoregressive_metrics(
        self,
        *,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor,
        masked_positions: torch.Tensor,
        labels: torch.Tensor,
        valid_labels: torch.Tensor,
    ) -> dict:
        if self.autoregressive_decoder is None:
            raise RuntimeError("The model is not using the autoregressive decoder")
        hidden = self.backbone(tokens, mask, type_ids)
        selected = hidden[masked_positions]
        positions = masked_positions.nonzero(as_tuple=False)[:, 1]
        selected_types = type_ids[masked_positions]
        return self.autoregressive_decoder.teacher_forced_metrics(
            context=selected,
            labels=labels,
            valid_labels=valid_labels,
            type_ids=selected_types,
            positions=positions,
            token_embedding=self.backbone.token_embedding,
        )

    def generate_masked_object_codes(
        self,
        *,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor,
        masked_positions: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.autoregressive_decoder is None:
            raise RuntimeError("The model is not using the autoregressive decoder")
        hidden = self.backbone(tokens, mask, type_ids)[masked_positions]
        selected_types = type_ids[masked_positions]
        object_types = self.autoregressive_decoder.object_type_ids
        is_object = torch.zeros_like(selected_types, dtype=torch.bool)
        for type_id in object_types:
            is_object |= selected_types.eq(type_id)
        if not is_object.all():
            raise ValueError(
                "generate_masked_object_codes only accepts masked physics-object "
                "positions"
            )
        return self.autoregressive_decoder.generate_object_codes(
            context=hidden,
            type_ids=selected_types,
            token_embedding=self.backbone.token_embedding,
            sample=sample,
            temperature=temperature,
            generator=generator,
        )


class LitGroupedMaskedSequenceModel(ScheduledOptimiserMixin, LightningModule):
    """Lightning wrapper for grouped object-level masked pretraining."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocab_size: int = 20000,
        max_quantizers: int = 8,
        quantizer_embedding_dim: int | None = None,
        num_type_ids: int = 12,
        mask_token_id: int = 3,
        pad_token_id: int = 0,
        mask_prob: float = 0.15,
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        quantizer_decoder: str = "parallel",
        token_vocabulary: dict | None = None,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.model = GroupedMaskedSequenceModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_quantizers=max_quantizers,
            quantizer_embedding_dim=quantizer_embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_type_ids=num_type_ids,
            pad_token_id=pad_token_id,
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
            quantizer_decoder=quantizer_decoder,
            token_vocabulary=token_vocabulary,
        )

    @property
    def backbone(self) -> GroupedSequenceBackbone:
        return self.model.backbone

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        original = batch[TOKENS_KEY]
        object_mask = batch[MASK_KEY].bool()
        type_ids = batch.get(TYPE_IDS_KEY)
        quantizer_mask = original.ne(self.pad_token_id)

        probabilities = torch.full(object_mask.shape, self.mask_prob, device=original.device)
        contains_special_token = ((original <= 3) & quantizer_mask).any(dim=-1)
        can_mask = object_mask & ~contains_special_token
        if self.model.quantizer_decoder == "autoregressive":
            if type_ids is None:
                raise RuntimeError(
                    "Autoregressive quantizer decoding requires token type IDs"
                )
            can_mask &= self.model.maskable_positions(type_ids)
        masked_positions = torch.bernoulli(probabilities).bool() & can_mask
        if not masked_positions.any():
            first_valid = can_mask.flatten().nonzero(as_tuple=False)
            if first_valid.numel() == 0:
                raise RuntimeError("Grouped token batch contains no maskable object positions")
            masked_positions.view(-1)[first_valid[0, 0]] = True

        tokens = original.clone()
        replace = masked_positions.unsqueeze(-1) & quantizer_mask
        tokens[replace] = self.mask_token_id
        labels = original[masked_positions]
        valid_labels = quantizer_mask[masked_positions]
        if self.model.quantizer_decoder == "parallel":
            logits = self.model.masked_logits(
                tokens, object_mask, type_ids, masked_positions
            )
            loss = F.cross_entropy(logits[valid_labels], labels[valid_labels])
            accuracy = (
                logits.argmax(dim=-1)[valid_labels] == labels[valid_labels]
            ).float().mean()
            quantizer_metrics = None
        else:
            metrics = self.model.autoregressive_metrics(
                tokens=tokens,
                mask=object_mask,
                type_ids=type_ids,
                masked_positions=masked_positions,
                labels=labels,
                valid_labels=valid_labels,
            )
            loss = metrics["loss"]
            accuracy = metrics["accuracy"]
            quantizer_metrics = metrics
        sync_dist = prefix == "valid"
        self.log(
            f"{prefix}/mask_acc",
            accuracy,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        self.log(
            f"{prefix}/total_loss",
            loss,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        if quantizer_metrics is not None:
            for quantizer, count in enumerate(
                quantizer_metrics["quantizer_counts"]
            ):
                denominator = count.clamp_min(1)
                self.log(
                    f"{prefix}/q{quantizer}_loss",
                    quantizer_metrics["quantizer_loss_sums"][quantizer]
                    / denominator,
                    sync_dist=sync_dist,
                )
                self.log(
                    f"{prefix}/q{quantizer}_acc",
                    quantizer_metrics["quantizer_correct"][quantizer].float()
                    / denominator,
                    sync_dist=sync_dist,
                )
        return loss

    def training_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "valid")
