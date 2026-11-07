"""ML-marked tests: tiny torch LoRA module and LeRobot normalization parity.

Skipped with a reason when torch/peft/LeRobot are absent.  Nothing here needs a
GPU or network access.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional .venv-ml torch environment is absent")
pytest.importorskip("peft", reason="optional .venv-ml PEFT package is absent")
pytest.importorskip("lerobot", reason="optional .venv-ml LeRobot environment is absent")

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from torch import nn  # noqa: E402

MEAN_STD_STATS = {
    "mean": np.array([1.0, -2.0, 0.5]),
    "std": np.array([0.5, 2.0, 1.5]),
}
QUANTILE_STATS = {
    "q01": np.array([-1.0, 0.0, 10.0]),
    "q99": np.array([1.0, 4.0, 14.0]),
}


class TinyTransformer(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x) + self.v_proj(x)) + self.k_proj(x))


def _lora_model():
    torch.manual_seed(0)
    base = TinyTransformer()
    originals = [(name, parameter) for name, parameter in base.named_parameters()]
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    return base, get_peft_model(base, config), originals


@pytest.mark.ml
def test_tiny_lora_trainable_fraction_and_frozen_base():
    _base, model, originals = _lora_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    fraction = trainable / total
    assert 0.0 < fraction < 0.25
    before = {name: parameter.detach().clone() for name, parameter in originals}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    inputs = torch.randn(4, 32)
    loss = model(inputs).pow(2).mean()
    loss.backward()
    optimizer.step()
    for name, parameter in originals:
        assert torch.equal(parameter.detach(), before[name]), f"frozen weight changed: {name}"
    print(f"tiny LoRA trainable fraction: {fraction:.6f}")


@pytest.mark.ml
def test_tiny_lora_save_and_load_round_trip(tmp_path):
    _base, model, _originals = _lora_model()
    model.eval()
    inputs = torch.randn(2, 32)
    expected = model(inputs).detach()
    model.save_pretrained(tmp_path / "adapter")
    # A LoRA checkpoint only carries the adapters: reload onto the same base
    # initialisation, exactly as a real base checkpoint would.
    torch.manual_seed(0)
    fresh = TinyTransformer()
    loaded = PeftModel.from_pretrained(fresh, tmp_path / "adapter", is_trainable=True)
    loaded.eval()
    actual = loaded(inputs).detach()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.ml
def test_our_normalization_matches_lerobot_on_the_same_fixture():
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    from sim2real.training.normalization import FeatureStats, Normalizer

    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    stats = {
        "action": {key: value.tolist() for key, value in MEAN_STD_STATS.items()},
        "observation.state": {key: value.tolist() for key, value in QUANTILE_STATS.items()},
    }
    step = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    values = np.array([[0.0, 1.0, 2.0], [3.0, -4.0, 0.5]])
    tensor = torch.as_tensor(values, dtype=torch.float32)

    differences = []
    for key, feature_type, our_stats in (
        ("action", FeatureType.ACTION, MEAN_STD_STATS),
        ("observation.state", FeatureType.STATE, QUANTILE_STATS),
    ):
        mode = "mean_std" if key == "action" else "quantile"
        ours = Normalizer(mode, FeatureStats.from_mapping(our_stats))
        forward_lerobot = step._apply_transform(tensor, key, feature_type).numpy()
        forward_ours = ours.apply(values)
        differences.append(float(np.max(np.abs(forward_lerobot - forward_ours))))

        normalized = torch.as_tensor(forward_ours, dtype=torch.float32)
        inverse_lerobot = step._apply_transform(normalized, key, feature_type, inverse=True).numpy()
        inverse_ours = ours.invert(forward_ours)
        differences.append(float(np.max(np.abs(inverse_lerobot - inverse_ours))))

    maximum = max(differences)
    print(f"max normalisation difference vs LeRobot: {maximum:.3e}")
    assert maximum < 1e-5
