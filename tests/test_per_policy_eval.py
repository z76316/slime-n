"""Tests for per-policy eval schema and resolver.

Covers `PolicyConfig.eval_datasets`, `EvalDatasetConfig.policies`,
top-level YAML parsing, the misplacement guard, the
`build_per_policy_eval_datasets` resolver (per-policy declarations +
--eval-config fan-out + legacy fallback), engine-required validator,
and the fail-fast guard that rejects per-policy declarations until
RolloutManager.eval reads them.

Pure-Python: no Ray, no GPUs, no Megatron. Config-layer only.

Run with:
    python -m pytest tests/test_per_policy_eval.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.policy_config import (
    PolicyConfig,
    _guard_eval_not_yet_wired_to_rollout,
    _validate_eval_datasets_require_sglang_engine,
    build_per_policy_eval_datasets,
    parse_policy_configs,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _yaml(tmp_path, policies):
    p = os.path.join(tmp_path, "config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"policies": policies}, f)
    return p


def _policy_entry(name="p", **kw):
    return {
        "name": name,
        "hf_checkpoint": "/x",
        "num_gpus_per_node": 1,
        "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
        **kw,
    }


def _policy(name="p", **kw):
    """Minimal PolicyConfig builder for resolver/guard tests."""
    return PolicyConfig(name=name, hf_checkpoint="/x", **kw)


def _eval_ds(name="aime", path="/root/aime.jsonl", **kw):
    return EvalDatasetConfig(name=name, path=path, rm_type="deepscaler", **kw)


def _base_args(eval_datasets=None, eval_interval=None):
    return argparse.Namespace(eval_datasets=eval_datasets, eval_interval=eval_interval)


# ── Schema ──────────────────────────────────────────────────────────────


def test_policy_config_carries_eval_datasets_field():
    cfg = _policy()
    assert cfg.eval_datasets is None


def test_eval_dataset_config_carries_policies_field():
    ds = EvalDatasetConfig(name="aime", path="/x", policies=["solver"])
    assert ds.policies == ["solver"]


# ── YAML round-trip ─────────────────────────────────────────────────────


def test_yaml_top_level_eval_datasets_round_trip(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            _policy_entry(
                eval_datasets=[
                    {"name": "aime", "path": "/root/aime.jsonl", "rm_type": "deepscaler"},
                ]
            )
        ],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.eval_datasets == [{"name": "aime", "path": "/root/aime.jsonl", "rm_type": "deepscaler"}]


def test_yaml_omitted_eval_datasets_default_to_none(tmp_path):
    cfg_path = _yaml(tmp_path, [_policy_entry()])
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.eval_datasets is None


# ── Misplacement guard ─────────────────────────────────────────────────


def test_eval_datasets_under_megatron_block_rejected(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [_policy_entry(megatron={"eval_datasets": [{"name": "aime", "path": "/x"}]})],
    )
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_eval_datasets_under_sglang_block_rejected(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "sglang": {
                    "eval_datasets": [{"name": "aime", "path": "/x"}],
                    "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


# ── Resolver: per-policy declarations ──────────────────────────────────


def test_form_a_per_policy_eval_datasets():
    cfgs = [
        _policy(name="solver", eval_datasets=[{"name": "aime", "path": "/a"}]),
        _policy(name="critic", eval_datasets=[{"name": "rubric", "path": "/r"}]),
    ]
    base = _base_args()
    out = build_per_policy_eval_datasets(cfgs, base)
    assert [d.name for d in out["solver"]] == ["aime"]
    assert [d.name for d in out["critic"]] == ["rubric"]


def test_form_a_strips_policies_fan_out_field():
    """If a user copy-pastes an --eval-config entry (with `policies:`)
    into a PolicyConfig.eval_datasets list, the resolver strips the
    fan-out field so EvalDatasetConfig(**entry) doesn't TypeError on
    an unrelated kwarg."""
    cfgs = [_policy(name="solver", eval_datasets=[{"name": "aime", "path": "/a", "policies": ["critic"]}])]
    out = build_per_policy_eval_datasets(cfgs, _base_args())
    assert [d.name for d in out["solver"]] == ["aime"]
    # Dataset belongs to solver (per-policy declaration); fan-out stripped.
    assert "critic" not in out or not out.get("critic")


# ── Resolver: --eval-config fan-out ────────────────────────────────────


def test_form_b_eval_config_policies_fan_out():
    cfgs = [_policy(name="solver"), _policy(name="critic")]
    aime = _eval_ds(name="aime", policies=["solver", "critic"])
    base = _base_args(eval_datasets=[aime])
    out = build_per_policy_eval_datasets(cfgs, base)
    assert [d.name for d in out["solver"]] == ["aime"]
    assert [d.name for d in out["critic"]] == ["aime"]


def test_form_b_dataset_without_policies_does_not_fan_out():
    """A dataset with no `policies:` field is not delivered to any policy
    via fan-out (it falls through to legacy fallback if nothing else
    declares a target)."""
    cfgs = [_policy(name="solver", trainable=True), _policy(name="critic")]
    bare = _eval_ds(name="aime", policies=None)
    base = _base_args(eval_datasets=[bare])
    out = build_per_policy_eval_datasets(cfgs, base)
    # Legacy fallback: first trainable-paired policy inherits.
    assert [d.name for d in out["solver"]] == ["aime"]
    assert out["critic"] == []


def test_form_b_unknown_policy_name_rejected():
    cfgs = [_policy(name="solver")]
    bad = _eval_ds(name="aime", policies=["nobody"])
    base = _base_args(eval_datasets=[bad])
    with pytest.raises(ValueError, match="not in policy_configs"):
        build_per_policy_eval_datasets(cfgs, base)


def test_form_b_policy_without_engine_rejected():
    cfgs = [
        _policy(name="solver"),
        _policy(name="critic", sglang_num_nodes=0),  # Megatron-only
    ]
    bad = _eval_ds(name="aime", policies=["critic"])
    base = _base_args(eval_datasets=[bad])
    with pytest.raises(ValueError, match="no SGLang engine"):
        build_per_policy_eval_datasets(cfgs, base)


# ── Resolver: per-policy vs fan-out precedence ─────────────────────────


def test_per_policy_decl_wins_over_fan_out_on_duplicate_name():
    """Solver declares aime on PolicyConfig; --eval-config also fans
    aime out to solver. The resolver dedupes by name within a policy's
    list — the per-policy entry is kept, the fan-out duplicate is
    discarded for solver. Other policies in the fan-out still
    receive aime."""
    cfgs = [_policy(name="solver"), _policy(name="critic")]
    aime_fan_out = _eval_ds(name="aime", policies=["solver", "critic"])
    base = _base_args(eval_datasets=[aime_fan_out])
    cfgs[0] = _policy(name="solver", eval_datasets=[{"name": "aime", "path": "/a"}])

    out = build_per_policy_eval_datasets(cfgs, base)

    # Solver: per-policy entry (path /a), not the fan-out (path /root/aime.jsonl).
    assert len(out["solver"]) == 1
    assert out["solver"][0].path == "/a"
    # Critic: fan-out entry (unaffected by solver's per-policy declaration).
    assert len(out["critic"]) == 1
    assert out["critic"][0].name == "aime"


# ── Resolver: legacy fallback ───────────────────────────────────────────


def test_legacy_fallback_first_trainable_paired():
    """When no per-policy declarations and no fan-out targets any
    policy, and global eval_datasets exist, the first trainable-paired
    (engine-hosting, trainable=True) policy inherits all global
    datasets."""
    cfgs = [
        _policy(name="teacher", trainable=False, sglang_num_nodes=0),
        _policy(name="solver", trainable=True),
        _policy(name="critic", trainable=True),
    ]
    aime = _eval_ds(name="aime")
    base = _base_args(eval_datasets=[aime])
    out = build_per_policy_eval_datasets(cfgs, base)
    assert [d.name for d in out["solver"]] == ["aime"]
    assert out["critic"] == []
    assert out["teacher"] == []


def test_no_eval_anywhere_returns_empty_map():
    cfgs = [_policy(name="solver"), _policy(name="critic")]
    out = build_per_policy_eval_datasets(cfgs, _base_args())
    assert out == {"solver": [], "critic": []}


# ── Engine-required validator ──────────────────────────────────────────


def test_validate_eval_datasets_require_sglang_engine_passes_when_engine():
    cfgs = [_policy(name="solver", eval_datasets=[{"name": "aime", "path": "/x"}])]
    _validate_eval_datasets_require_sglang_engine(cfgs)  # no raise


def test_validate_eval_datasets_require_sglang_engine_rejects_megatron_only():
    cfgs = [
        _policy(
            name="critic",
            eval_datasets=[{"name": "aime", "path": "/x"}],
            sglang_num_nodes=0,
        )
    ]
    with pytest.raises(ValueError, match="no SGLang engine"):
        _validate_eval_datasets_require_sglang_engine(cfgs)


# ── Fail-fast guard ────────────────────────────────────────────────────


def test_guard_passes_when_no_per_policy_eval():
    cfgs = [_policy(name="solver"), _policy(name="critic")]
    per_policy = {"solver": [], "critic": []}
    _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)  # no raise


def test_guard_passes_on_legacy_fallback_single_target():
    """Legacy fallback (no per-policy declarations, no fan-out) sends
    all global eval datasets to the first trainable-paired policy.
    That's not "per-policy declaration" — it's the existing
    single-engine path just stored in the new map shape. Guard should
    NOT fire."""
    cfgs = [_policy(name="solver", trainable=True), _policy(name="critic", trainable=True)]
    per_policy = {"solver": [_eval_ds(name="aime")], "critic": []}
    _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)  # no raise


def test_guard_rejects_form_a_declaration():
    cfgs = [_policy(name="solver", eval_datasets=[{"name": "aime", "path": "/x"}])]
    per_policy = {"solver": [_eval_ds()]}
    with pytest.raises(NotImplementedError, match="not yet wired"):
        _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)


def test_guard_rejects_form_b_multi_target():
    """Fan-out that sends one dataset to multiple policies → guard fires."""
    cfgs = [_policy(name="solver", trainable=True), _policy(name="critic", trainable=True)]
    per_policy = {"solver": [_eval_ds()], "critic": [_eval_ds()]}
    with pytest.raises(NotImplementedError, match="not yet wired"):
        _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)


def test_guard_rejects_non_default_target():
    """Fan-out targeting a non-default (not first trainable-paired)
    policy → guard fires even if only one policy is targeted."""
    cfgs = [_policy(name="solver", trainable=True), _policy(name="critic", trainable=True)]
    per_policy = {"solver": [], "critic": [_eval_ds()]}
    with pytest.raises(NotImplementedError, match="not yet wired"):
        _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)


def test_guard_rejects_explicit_fan_out_to_default_policy():
    """An --eval-config entry with `policies: [<first_trainable_paired>]`
    produces a resolved map identical to legacy fallback (single entry
    on the default policy), but the user's explicit fan-out intent must
    still be rejected — RolloutManager.eval has no per-policy iteration
    yet, so it would silently log under `eval/<dataset>` instead of
    `eval/<policy>/<dataset>`. The signal lives on base_args, not on
    the resolved map."""
    cfgs = [_policy(name="solver", trainable=True)]
    explicit = _eval_ds(name="aime", policies=["solver"])
    base = _base_args(eval_datasets=[explicit])
    per_policy = {"solver": [explicit]}
    with pytest.raises(NotImplementedError, match="fan_out_datasets"):
        _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy, base)


def test_guard_passes_without_base_args_for_legacy_callers():
    """Pre-this-guard callers that don't pass base_args still get the
    legacy three-signal detection (no source-side fan-out check)."""
    cfgs = [_policy(name="solver", trainable=True)]
    per_policy = {"solver": [_eval_ds()]}
    # No base_args, no per-policy intent on PolicyConfig → guard passes
    # (this is the legacy single-target case).
    _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy)
    _guard_eval_not_yet_wired_to_rollout(cfgs, per_policy, base_args=None)
