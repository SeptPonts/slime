from __future__ import annotations

import asyncio
import base64
import sys
import types
from pathlib import Path
from types import SimpleNamespace

NUM_GPUS = 0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "fully_async"))

if "ray" not in sys.modules:
    ray_mod = types.ModuleType("ray")
    ray_mod._private = types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_mod

if "sglang_router" not in sys.modules:
    sglang_router_mod = types.ModuleType("sglang_router")
    sglang_router_mod.__version__ = "0.2.3"
    sys.modules["sglang_router"] = sglang_router_mod

if "transformers" not in sys.modules:
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
    )
    transformers_mod.AutoProcessor = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
    )
    transformers_mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    transformers_mod.ProcessorMixin = type("ProcessorMixin", (), {})
    sys.modules["transformers"] = transformers_mod

if "torch" not in sys.modules:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = type("dtype", (), {})
    torch_mod.Size = tuple
    torch_distributed_mod = types.ModuleType("torch.distributed")
    torch_distributed_mod.is_initialized = lambda: False
    torch_distributed_mod.get_rank = lambda: 0
    torch_mod.distributed = torch_distributed_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = torch_distributed_mod

if "pybase64" not in sys.modules:
    pybase64_mod = types.ModuleType("pybase64")
    pybase64_mod.b64decode = base64.b64decode
    sys.modules["pybase64"] = pybase64_mod

if "tqdm" not in sys.modules:
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda *args, **kwargs: types.SimpleNamespace(update=lambda *a, **k: None, close=lambda: None)
    sys.modules["tqdm"] = tqdm_mod

if "pylatexenc" not in sys.modules:
    pylatexenc_mod = types.ModuleType("pylatexenc")
    latex2text_mod = types.ModuleType("latex2text")
    latex2text_mod.LatexNodes2Text = type("LatexNodes2Text", (), {"latex_to_text": lambda self, value: value})
    pylatexenc_mod.latex2text = latex2text_mod
    sys.modules["pylatexenc"] = pylatexenc_mod

import fully_async_rollout as fully_async_mod
from fully_async_rollout import generate_rollout_async
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample
from staleness import StalenessTracker, evaluate_group_staleness


def make_args(**overrides):
    defaults = {
        "rollout_global_dataset": True,
        "rollout_batch_size": 2,
        "dynamic_sampling_filter_path": None,
        "rollout_sample_filter_path": None,
        "rollout_all_samples_process_path": None,
        "fully_async_max_staleness": None,
        "fully_async_drop_unknown_version": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_sample(index: int, *, weight_versions=None, reward: float = 1.0) -> Sample:
    return Sample(
        index=index,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        tokens=[1000 + index],
        response_length=1,
        reward=reward,
        weight_versions=list(weight_versions or []),
        status=Sample.Status.COMPLETED,
    )


class FakeDataBuffer:
    def __init__(self):
        self.requeued = []

    def add_samples(self, samples):
        self.requeued.extend(samples)


class FakeWorker:
    def __init__(self, completed_groups):
        self._completed_groups = [completed_groups]
        self.staleness_tracker = StalenessTracker()

    def get_completed_groups(self):
        if self._completed_groups:
            return self._completed_groups.pop(0)
        return []

    def get_queue_size(self) -> int:
        return 0


def test_staleness_evaluator_handles_unknown_and_stale_versions():
    tracker = StalenessTracker()

    unknown_group = [make_sample(0, weight_versions=["invalid"])]
    unknown_decision = evaluate_group_staleness(
        unknown_group,
        tracker,
        max_staleness=1,
        drop_unknown_version=True,
    )
    assert not unknown_decision.keep
    assert unknown_decision.reason == "unknown_version"
    assert tracker.latest_seen_weight_version is None

    fresh_group = [make_sample(1, weight_versions=["7", "5"])]
    fresh_decision = evaluate_group_staleness(
        fresh_group,
        tracker,
        max_staleness=1,
        drop_unknown_version=True,
    )
    assert fresh_decision.keep
    assert fresh_decision.group_weight_version == 7
    assert tracker.latest_seen_weight_version == 7

    stale_group = [make_sample(2, weight_versions=["5"])]
    stale_decision = evaluate_group_staleness(
        stale_group,
        tracker,
        max_staleness=1,
        drop_unknown_version=True,
    )
    assert not stale_decision.keep
    assert stale_decision.reason == "stale"


def test_generate_rollout_async_preserves_backward_compatibility_without_staleness(monkeypatch):
    accepted_group = [make_sample(0)]
    fake_worker = FakeWorker([(0, accepted_group)])
    fake_data_buffer = FakeDataBuffer()

    monkeypatch.setattr(fully_async_mod, "get_global_worker", lambda args, data_buffer: fake_worker)

    output = asyncio.run(generate_rollout_async(make_args(rollout_batch_size=1), 0, fake_data_buffer))

    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples == [accepted_group]
    assert output.metrics["fully_async/stale_drop_count"] == 0
    assert output.metrics["fully_async/unknown_version_drop_count"] == 0
    assert output.metrics["fully_async/max_seen_weight_version"] == -1


def test_generate_rollout_async_applies_hooks_and_stale_filtering(monkeypatch):
    calls = {"all_samples": None, "data_source": None}

    def dynamic_filter(args, group):
        if group[0].index == 20:
            return DynamicFilterOutput(keep=False, reason="manual")
        return DynamicFilterOutput(keep=True)

    def rollout_sample_filter(args, groups):
        groups[0][0].remove_sample = True

    def process_all_samples(args, all_samples, data_source):
        calls["all_samples"] = all_samples
        calls["data_source"] = data_source

    accepted_old = [make_sample(0, weight_versions=["5"], reward=1.0)]
    stale_group = [make_sample(10, weight_versions=["3"], reward=1.0)]
    dynamic_drop_group = [make_sample(20, weight_versions=["6"], reward=1.0)]
    accepted_new = [make_sample(30, weight_versions=["6"], reward=1.0)]

    fake_worker = FakeWorker(
        [
            (0, accepted_old),
            (1, stale_group),
            (2, dynamic_drop_group),
            (3, accepted_new),
        ]
    )
    fake_data_buffer = FakeDataBuffer()
    mapping = {
        "dynamic.filter": dynamic_filter,
        "sample.filter": rollout_sample_filter,
        "all.process": process_all_samples,
    }

    monkeypatch.setattr(fully_async_mod, "get_global_worker", lambda args, data_buffer: fake_worker)
    monkeypatch.setattr(fully_async_mod, "load_function", lambda path: mapping[path])

    args = make_args(
        rollout_batch_size=2,
        dynamic_sampling_filter_path="dynamic.filter",
        rollout_sample_filter_path="sample.filter",
        rollout_all_samples_process_path="all.process",
        fully_async_max_staleness=1,
        fully_async_drop_unknown_version=True,
    )
    output = asyncio.run(generate_rollout_async(args, 0, fake_data_buffer))

    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples == [accepted_old, accepted_new]
    assert output.samples[0][0].remove_sample is True
    assert output.metrics["rollout/dynamic_filter/drop_manual"] == 1
    assert output.metrics["fully_async/stale_drop_count"] == 1
    assert output.metrics["fully_async/stale_drop_ratio"] == 0.25
    assert output.metrics["fully_async/unknown_version_drop_count"] == 0
    assert output.metrics["fully_async/max_seen_weight_version"] == 6
    assert output.metrics["fully_async/min_accepted_weight_version"] == 5
    assert calls["all_samples"] == [accepted_old, stale_group, dynamic_drop_group, accepted_new]
    assert calls["data_source"] is fake_data_buffer
