"""Policy adapters used by the trainer.

``FakePolicyAdapter`` is NumPy-only so the training loop, checkpointing, and
resume behaviour are testable in the ROS-free core environment.
``SmolVLAAdapter`` imports torch/LeRobot/PEFT lazily and is guarded by the ``ml``
marker.  ``OpenVLAAdapter`` is an explicit, unvalidated stub: it raises
``AdapterUnavailable`` instead of pretending to run a 7B Llama-derived model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .lora import (
    DEFAULT_SMOLVLA_TARGET_MODULES,
    LoraTargetError,
    adapted_scope,
    group_by_type,
    resolve_target_modules,
    trainable_parameter_report,
)


class AdapterError(RuntimeError):
    """Base error for adapter problems."""


class AdapterUnavailable(RuntimeError):
    """Raised when an optional ML dependency or validated path is unavailable."""


@runtime_checkable
class PolicyAdapter(Protocol):
    name: str

    def forward_loss(self, batch: dict[str, Any]) -> Any: ...

    def backward(self, loss: Any) -> None: ...

    def optimizer_step(self, lr: float) -> None: ...

    def zero_grad(self) -> None: ...

    def trainable_parameters(self) -> int: ...

    def total_parameters(self) -> int: ...

    def save_adapter(self, path: str | Path) -> None: ...

    def load_adapter(self, path: str | Path) -> None: ...

    def state_dict_hash(self) -> str: ...


def _finite(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    number = float(value)
    if not np.isfinite(number):
        raise AdapterError("loss is not finite")
    return number


class FakePolicyAdapter:
    """A small linear model trained by manual gradient descent.

    ``forward`` is ``y = x @ W + b`` with mean-squared error.  It is deliberately
    tiny so the trainer's determinism, resume, and NaN-abort paths are exactly
    reproducible without torch.
    """

    name = "fake_linear"

    def __init__(self, input_dim: int, output_dim: int, *, seed: int = 0, scale: float = 0.05):
        if input_dim < 1 or output_dim < 1:
            raise AdapterError("input_dim and output_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, scale, size=(self.input_dim, self.output_dim))
        self.bias = np.zeros(self.output_dim, dtype=np.float64)
        self._grad_weights = np.zeros_like(self.weights)
        self._grad_bias = np.zeros_like(self.bias)
        self._last_grad_weights = np.zeros_like(self.weights)
        self._last_grad_bias = np.zeros_like(self.bias)
        self._size = self.weights.size + self.bias.size

    def _split(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if "inputs" not in batch or "targets" not in batch:
            raise AdapterError("batch must contain 'inputs' and 'targets'")
        inputs = np.asarray(batch["inputs"], dtype=np.float64)
        targets = np.asarray(batch["targets"], dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise AdapterError(f"inputs: expected (B, {self.input_dim})")
        if targets.shape != (inputs.shape[0], self.output_dim):
            raise AdapterError(f"targets: expected {(inputs.shape[0], self.output_dim)}")
        if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets)):
            raise AdapterError("batch contains NaN or infinity")
        return inputs, targets

    def forward_loss(self, batch: dict[str, Any]) -> float:
        inputs, targets = self._split(batch)
        predictions = inputs @ self.weights + self.bias
        residual = predictions - targets
        batch_size = inputs.shape[0]
        loss = float(np.mean(residual**2))
        # Mean-squared error gradients, accumulated by ``backward``.
        self._last_grad_weights = (2.0 / (batch_size * self.output_dim)) * (inputs.T @ residual)
        self._last_grad_bias = (2.0 / (batch_size * self.output_dim)) * residual.sum(axis=0)
        if not np.isfinite(loss):
            raise AdapterError("loss is not finite")
        return loss

    def backward(self, loss: Any) -> None:
        if not np.isfinite(_finite(loss)):
            raise AdapterError("loss is not finite")
        self._grad_weights += self._last_grad_weights
        self._grad_bias += self._last_grad_bias

    def optimizer_step(self, lr: float) -> None:
        if not np.isfinite(lr) or lr <= 0:
            raise AdapterError("lr must be finite and > 0")
        self.weights -= lr * self._grad_weights
        self.bias -= lr * self._grad_bias
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise AdapterError("adapter weights became non-finite")
        self.zero_grad()

    def zero_grad(self) -> None:
        self._grad_weights.fill(0.0)
        self._grad_bias.fill(0.0)

    def trainable_parameters(self) -> int:
        return int(self._size)

    def total_parameters(self) -> int:
        return int(self._size)

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy(), "bias": self.bias.copy()}

    def _hash_arrays(self, arrays: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def state_dict_hash(self) -> str:
        return self._hash_arrays(self.state_dict())

    def save_adapter(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        arrays = self.state_dict()
        np.savez(target / "fake_adapter.npz", **arrays)
        (target / "fake_adapter.json").write_text(
            json.dumps(
                {
                    "kind": self.name,
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "state_dict_hash": self._hash_arrays(arrays),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_adapter(self, path: str | Path) -> None:
        target = Path(path)
        metadata_path = target / "fake_adapter.json"
        array_path = target / "fake_adapter.npz"
        if not metadata_path.is_file() or not array_path.is_file():
            raise AdapterError(f"{target}: fake adapter files not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("kind") != self.name:
            raise AdapterError(f"{target}: not a {self.name} checkpoint")
        if (
            int(metadata["input_dim"]) != self.input_dim
            or int(metadata["output_dim"]) != self.output_dim
        ):
            raise AdapterError(f"{target}: adapter dimensions do not match this model")
        with np.load(array_path) as arrays:
            weights = np.asarray(arrays["weights"], dtype=np.float64)
            bias = np.asarray(arrays["bias"], dtype=np.float64)
        if weights.shape != self.weights.shape or bias.shape != self.bias.shape:
            raise AdapterError(f"{target}: adapter array shapes do not match")
        self.weights = weights
        self.bias = bias
        if metadata.get("state_dict_hash") != self.state_dict_hash():
            raise AdapterError(f"{target}: adapter hash mismatch")


@dataclass
class LoraTargetReport:
    resolved: list[str]
    groups: dict[str, list[str]]
    scope_counts: dict[str, int]


class SmolVLAAdapter:
    """LeRobot SmolVLA + PEFT LoRA, imported lazily.

    The base model, its revision, chunk size, and LoRA targets all come from the
    validated config.  Target modules are resolved against the loaded model and
    a pattern that matches nothing is a hard error.
    """

    name = "smolvla_lora"

    def __init__(
        self,
        config: Any,
        *,
        device: str | None = None,
        dataset_stats: Any = None,
        dataset_meta: Any = None,
    ):
        try:
            import torch  # noqa: F401
            from lerobot.policies.smolvla import SmolVLAConfig
        except ImportError as exc:  # pragma: no cover - depends on .venv-ml
            raise AdapterUnavailable(
                "SmolVLAAdapter requires torch, LeRobot and PEFT in .venv-ml; "
                f"missing dependency: {exc}. See docs/TRAINING.md."
            ) from exc
        self.config = config
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config.training.deterministic:
            torch.manual_seed(config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.training.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        overrides = [
            f"--chunk_size={config.training.chunk_size}",
            f"--n_action_steps={config.training.n_action_steps}",
            f"--max_state_dim={config.model.max_state_dim}",
            f"--max_action_dim={config.model.max_action_dim}",
            f"--empty_cameras={config.model.empty_cameras}",
        ]
        base_config = SmolVLAConfig.from_pretrained(
            config.model.checkpoint,
            revision=config.model.revision,
            cli_overrides=overrides,
        )
        base_config.device = self.device
        base_config.pretrained_path = config.model.checkpoint
        base_config.pretrained_revision = config.model.revision
        if dataset_meta is not None:
            from lerobot.policies import make_policy

            self.policy = make_policy(
                cfg=base_config,
                ds_meta=dataset_meta,
                rename_map=config.data.rename_map,
            )
        else:
            from lerobot.policies.smolvla import SmolVLAPolicy

            self.policy = SmolVLAPolicy.from_pretrained(
                config.model.checkpoint,
                config=base_config,
                revision=config.model.revision,
            ).to(self.device)
        # Keep the base policy config for processor construction: after a PEFT
        # wrap, ``policy.config`` is a PeftConfig, not a SmolVLAConfig.
        self.base_policy_config = base_config
        self.preprocessor = None
        self.postprocessor = None
        if dataset_stats is not None:
            # Use LeRobot's official factory with the same overrides as
            # lerobot_train.py: rename the dataset cameras to the model's keys,
            # and use the dataset's authoritative stats (not the base model's).
            from lerobot.policies import make_pre_post_processors

            features = {
                **self.base_policy_config.input_features,
                **self.base_policy_config.output_features,
            }
            norm_map = self.base_policy_config.normalization_mapping
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.base_policy_config,
                pretrained_path=config.model.checkpoint,
                pretrained_revision=config.model.revision,
                dataset_stats=dataset_stats,
                preprocessor_overrides={
                    "device_processor": {"device": self.device},
                    "normalizer_processor": {
                        "features": features,
                        "norm_map": norm_map,
                        "stats": dataset_stats,
                    },
                    "rename_observations_processor": {"rename_map": dict(config.data.rename_map)},
                },
                postprocessor_overrides={
                    "unnormalizer_processor": {
                        "features": self.base_policy_config.output_features,
                        "norm_map": norm_map,
                        "stats": dataset_stats,
                    },
                },
            )
        self.target_report: LoraTargetReport | None = None
        if config.lora.enabled:
            self.policy = self._wrap_lora()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.policy.parameters() if parameter.requires_grad],
            lr=config.training.lr,
        )

    def _wrap_lora(self) -> Any:
        # Resolve first: never trust a name that does not exist in this model.
        named_types = {name: type(module).__name__ for name, module in self.policy.named_modules()}
        report = self.resolve_target_modules(list(named_types))
        overrides = {
            "method_type": self.config.lora.method_type,
            # PEFT turns a list into a set and stops treating it as a regex.
            # Hand it the exact names resolved against this model instead.
            "target_modules": list(report.resolved),
            "r": self.config.lora.r,
            "lora_alpha": self.config.lora.alpha,
            "lora_dropout": self.config.lora.dropout,
            "bias": self.config.lora.bias,
            "init_type": self.config.lora.init_type,
        }
        return self.policy.wrap_with_peft(peft_cli_overrides=overrides)

    def resolve_target_modules(self, module_names: list[str] | None = None) -> LoraTargetReport:
        if module_names is None and self.target_report is not None:
            return self.target_report
        patterns = list(self.config.lora.target_modules) or list(DEFAULT_SMOLVLA_TARGET_MODULES)
        if module_names is None:
            module_names = [name for name, _module in self.policy.named_modules()]
        resolved = resolve_target_modules(patterns, module_names)
        named_types = {name: type(module).__name__ for name, module in self.policy.named_modules()}
        groups = group_by_type(
            (name, named_types[name]) for name in resolved if name in named_types
        )
        scope_counts: dict[str, int] = {}
        for name in resolved:
            scope = adapted_scope(name)
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        self.target_report = LoraTargetReport(
            resolved=resolved, groups=groups, scope_counts=scope_counts
        )
        return self.target_report

    def parameter_report(self) -> dict[str, float | int]:
        counts = {
            name: (int(parameter.numel()), bool(parameter.requires_grad))
            for name, parameter in self.policy.named_parameters()
        }
        return trainable_parameter_report(counts)

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.preprocessor is None:
            raise AdapterError("preprocessor is not configured (dataset stats were not supplied)")
        return self.preprocessor(batch)

    def forward_loss(self, batch: dict[str, Any]) -> Any:
        result = self.policy.forward(batch)
        if isinstance(result, tuple):
            loss = result[0]
        elif isinstance(result, dict):
            loss = result["loss"]
        else:
            loss = result
        return loss

    def backward(self, loss: Any) -> None:
        loss.backward()

    def optimizer_step(self, lr: float) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.policy.parameters() if p.requires_grad))

    def total_parameters(self) -> int:
        return int(sum(p.numel() for p in self.policy.parameters()))

    def state_dict_hash(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.policy.named_parameters()):
            if not parameter.requires_grad:
                continue
            digest.update(name.encode())
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(array.tobytes())
        return digest.hexdigest()

    def save_adapter(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(target)
        if self.target_report is not None:
            (target / "lora_targets.json").write_text(
                json.dumps(
                    {
                        "resolved": self.target_report.resolved,
                        "scope_counts": self.target_report.scope_counts,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def load_adapter(self, path: str | Path) -> None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - depends on .venv-ml
            raise AdapterUnavailable("load_adapter requires peft") from exc
        if not isinstance(self.policy, PeftModel):
            raise AdapterError("load_adapter requires a PEFT-wrapped policy")
        # Load into the existing wrapper. Re-wrapping a PeftModel would add a
        # second ``base_model.model.`` prefix and the weights would not match.
        self.policy.load_adapter(str(path), adapter_name="trained", is_trainable=True)
        self.policy.set_adapter("trained")


class OpenVLAAdapter:
    """Explicitly unvalidated OpenVLA-7B stub.

    The config file exists so the family is documented, not because inference or
    fine-tuning was verified.  This adapter never runs.
    """

    name = "openvla7b"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AdapterUnavailable(
            "OpenVLA-7B is not validated in Task 8: license (Llama-derived) and "
            "VRAM requirements must be verified first. Use family: smolvla."
        )

    def forward_loss(self, batch: dict[str, Any]) -> Any:  # pragma: no cover - always raises
        raise AdapterUnavailable("OpenVLA is not validated")

    def backward(self, loss: Any) -> None:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def optimizer_step(self, lr: float) -> None:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def zero_grad(self) -> None:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def trainable_parameters(self) -> int:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def total_parameters(self) -> int:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def save_adapter(self, path: str | Path) -> None:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def load_adapter(self, path: str | Path) -> None:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")

    def state_dict_hash(self) -> str:  # pragma: no cover
        raise AdapterUnavailable("OpenVLA is not validated")


def make_adapter(
    config: Any, *, fake: bool = False, dataset_stats: Any = None, dataset_meta: Any = None
) -> PolicyAdapter:
    if fake:
        return FakePolicyAdapter(8, 8, seed=config.training.seed)
    if config.model.family == "smolvla":
        return SmolVLAAdapter(config, dataset_stats=dataset_stats, dataset_meta=dataset_meta)
    if config.model.family == "openvla7b":
        return OpenVLAAdapter()
    raise AdapterUnavailable(f"unknown model family {config.model.family!r}")


__all__ = [
    "AdapterError",
    "AdapterUnavailable",
    "FakePolicyAdapter",
    "LoraTargetError",
    "LoraTargetReport",
    "OpenVLAAdapter",
    "PolicyAdapter",
    "SmolVLAAdapter",
    "make_adapter",
]
