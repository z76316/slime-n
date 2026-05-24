"""Phase A tests for per-policy input_key / label_key / apply_chat_template.

Covers schema, projection (override-or-inherit), and the shared-buffer
data-view consistency validator. Phase B (data source row-backed
materialization + multi-view fetch) gets its own test module.

Pure-Python: no Ray, no GPUs, no Megatron. Config-layer only.

Run with:
    python -m pytest tests/test_per_policy_input_keys.py -v
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

from slime.utils.policy_config import (
    POLICY_DATA_VIEW_KEYS,
    PolicyConfig,
    PolicyDataView,
    _build_policy_data_views,
    _guard_data_views_not_yet_wired_to_rollout,
    _validate_shared_buffer_data_view_consistency,
    config_to_namespace,
    parse_policy_configs,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _yaml(tmp_path, policies):
    p = os.path.join(tmp_path, "config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"policies": policies}, f)
    return p


def _policy(name="p", **kw):
    """Build a minimal valid YAML policy entry."""
    return {
        "name": name,
        "hf_checkpoint": "/x",
        "num_gpus_per_node": 1,
        "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
        **kw,
    }


def _base_args(**kw):
    """Build a CLI-globals stand-in for tests."""
    defaults = {"input_key": "prompt", "label_key": None, "apply_chat_template": False}
    defaults.update(kw)
    return argparse.Namespace(**defaults)


# ── Schema ──────────────────────────────────────────────────────────────


def test_policy_config_carries_data_view_fields():
    """PolicyConfig has the three nullable data-view fields with None defaults."""
    cfg = PolicyConfig(name="p", hf_checkpoint="/x")
    assert cfg.input_key is None
    assert cfg.label_key is None
    assert cfg.apply_chat_template is None


def test_policy_data_view_keys_constant():
    assert POLICY_DATA_VIEW_KEYS == ("input_key", "label_key", "apply_chat_template")


# ── Parsing: top-level YAML keys round-trip into PolicyConfig ───────────


def test_yaml_top_level_keys_round_trip(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [_policy(input_key="solver_prompt", label_key="label", apply_chat_template=True)],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.input_key == "solver_prompt"
    assert cfg.label_key == "label"
    assert cfg.apply_chat_template is True


def test_yaml_omitted_keys_default_to_none(tmp_path):
    cfg_path = _yaml(tmp_path, [_policy()])
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.input_key is None
    assert cfg.label_key is None
    assert cfg.apply_chat_template is None


# ── Projection: config_to_namespace override-or-inherit ─────────────────


def test_policy_config_input_key_overrides_inherited():
    """Set input_key on the policy → ns.input_key reflects the override.
    Leave it None → ns.input_key falls back to base_args."""
    base = _base_args(input_key="prompt")
    overridden = config_to_namespace(
        PolicyConfig(name="p", hf_checkpoint="/x", input_key="solver_prompt"),
        base,
    )
    inherited = config_to_namespace(PolicyConfig(name="q", hf_checkpoint="/x"), base)
    assert overridden.input_key == "solver_prompt"
    assert inherited.input_key == "prompt"


def test_policy_config_omitted_key_does_not_clobber_global():
    """Regression: with the dataclass field default None, the generic
    dataclass copy in config_to_namespace must SKIP the data-view keys.
    Otherwise an omitted YAML field would overwrite base_args.input_key
    with None instead of inheriting."""
    base = _base_args(input_key="prompt", label_key="label", apply_chat_template=True)
    ns = config_to_namespace(PolicyConfig(name="p", hf_checkpoint="/x"), base)
    assert ns.input_key == "prompt"
    assert ns.label_key == "label"
    assert ns.apply_chat_template is True


def test_label_key_override():
    base = _base_args(input_key="prompt", label_key="label")
    ns = config_to_namespace(
        PolicyConfig(name="p", hf_checkpoint="/x", label_key="critic_label"),
        base,
    )
    assert ns.label_key == "critic_label"


def test_apply_chat_template_can_be_disabled_per_policy():
    """A policy can flip apply_chat_template=False even when the global is True."""
    base = _base_args(apply_chat_template=True)
    ns = config_to_namespace(
        PolicyConfig(name="p", hf_checkpoint="/x", apply_chat_template=False),
        base,
    )
    assert ns.apply_chat_template is False


# ── Data-view builder ───────────────────────────────────────────────────


def test_build_policy_data_views_resolves_override_and_inherit():
    base = _base_args(input_key="global_prompt", label_key="label", apply_chat_template=True)
    cfgs = [
        PolicyConfig(name="solver", hf_checkpoint="/x", input_key="solver_prompt"),
        PolicyConfig(name="critic", hf_checkpoint="/x"),  # all inherited
    ]
    views = _build_policy_data_views(cfgs, base)
    assert views["solver"].input_key == "solver_prompt"
    assert views["solver"].declared_keys == frozenset({"input_key"})
    assert views["critic"].input_key == "global_prompt"
    assert views["critic"].declared_keys == frozenset()


def test_build_policy_data_views_includes_frozen_and_sglang_only():
    """Data views must cover every policy, including frozen producers and
    SGLang-only judges — not just trainable-paired ones."""
    base = _base_args()
    cfgs = [
        PolicyConfig(name="actor", hf_checkpoint="/x"),  # trainable paired
        PolicyConfig(name="teacher", hf_checkpoint="/x", trainable=False, sglang_num_nodes=0),  # frozen Megatron
        PolicyConfig(name="judge", hf_checkpoint="/x", megatron_num_nodes=0),  # SGLang-only
    ]
    views = _build_policy_data_views(cfgs, base)
    assert set(views.keys()) == {"actor", "teacher", "judge"}


def test_data_view_apply_chat_template_defaults_false_when_both_unset():
    """If neither override nor base_args carries apply_chat_template,
    the resolved value is False (matches the CLI argparse default)."""
    base = argparse.Namespace(input_key=None, label_key=None)  # no apply_chat_template attr
    views = _build_policy_data_views([PolicyConfig(name="p", hf_checkpoint="/x")], base)
    assert views["p"].apply_chat_template is False


# ── Shared-buffer data-view validator ───────────────────────────────────


def test_shared_buffer_rejects_mixed_effective_input_key():
    """Two shared-buffer policies that resolve to different input_keys must raise."""
    base = _base_args(input_key="prompt")
    cfgs = [
        PolicyConfig(name="a", hf_checkpoint="/x", buffer_mode="shared", input_key="x_prompt"),
        PolicyConfig(name="b", hf_checkpoint="/x", buffer_mode="shared", input_key="y_prompt"),
    ]
    with pytest.raises(ValueError, match="resolved 'input_key'"):
        _validate_shared_buffer_data_view_consistency(cfgs, base)


def test_shared_buffer_legacy_inherited_keys_allowed():
    """Backward compatibility: two shared-buffer policies that BOTH omit
    per-policy data keys inherit the same CLI global and pass."""
    base = _base_args(input_key="prompt", label_key="label", apply_chat_template=True)
    cfgs = [
        PolicyConfig(name="a", hf_checkpoint="/x", buffer_mode="shared"),
        PolicyConfig(name="b", hf_checkpoint="/x", buffer_mode="shared"),
    ]
    # Should not raise.
    _validate_shared_buffer_data_view_consistency(cfgs, base)


def test_shared_buffer_override_equals_global_allowed():
    """Edge case: one sibling overrides input_key to the same value the
    other inherits. Resolved values match; validator must not raise."""
    base = _base_args(input_key="prompt")
    cfgs = [
        PolicyConfig(name="a", hf_checkpoint="/x", buffer_mode="shared"),
        PolicyConfig(name="b", hf_checkpoint="/x", buffer_mode="shared", input_key="prompt"),
    ]
    _validate_shared_buffer_data_view_consistency(cfgs, base)  # no raise


def test_shared_buffer_mixed_apply_chat_template_raises():
    base = _base_args(apply_chat_template=True)
    cfgs = [
        PolicyConfig(name="a", hf_checkpoint="/x", buffer_mode="shared"),
        PolicyConfig(
            name="b",
            hf_checkpoint="/x",
            buffer_mode="shared",
            apply_chat_template=False,
        ),
    ]
    with pytest.raises(ValueError, match="resolved 'apply_chat_template'"):
        _validate_shared_buffer_data_view_consistency(cfgs, base)


def test_shared_buffer_with_split_siblings_is_unaffected():
    """Validator only inspects buffer_mode=shared policies; split-buffer
    siblings with different keys are fine."""
    base = _base_args()
    cfgs = [
        PolicyConfig(name="solver", hf_checkpoint="/x", input_key="solver_prompt"),
        PolicyConfig(name="critic", hf_checkpoint="/x", input_key="critic_prompt"),
    ]
    # buffer_mode defaults to "split"; validator is a no-op.
    _validate_shared_buffer_data_view_consistency(cfgs, base)


def test_shared_buffer_single_policy_short_circuits():
    """One shared-buffer policy alone has nothing to disagree with."""
    base = _base_args()
    cfgs = [PolicyConfig(name="solo", hf_checkpoint="/x", buffer_mode="shared")]
    _validate_shared_buffer_data_view_consistency(cfgs, base)


# ── PolicyDataView dataclass ────────────────────────────────────────────


def test_policy_data_view_is_frozen():
    v = PolicyDataView(input_key="p", label_key="l", apply_chat_template=True, declared_keys=frozenset())
    with pytest.raises(dataclasses_FrozenInstanceError()):
        v.input_key = "other"


def dataclasses_FrozenInstanceError():
    import dataclasses

    return dataclasses.FrozenInstanceError


# ── Misplaced under `megatron:` rejected (P3) ───────────────────────────


def test_data_view_keys_under_megatron_block_rejected(tmp_path):
    """The data-view fields are top-level policy entries; putting them
    under `megatron:` would silently shadow a top-level declaration
    because the post-`megatron:` merge order. Parser must reject."""
    cfg_path = _yaml(tmp_path, [_policy(megatron={"input_key": "wrong_place"})])
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_label_key_under_megatron_block_rejected(tmp_path):
    cfg_path = _yaml(tmp_path, [_policy(megatron={"label_key": "wrong"})])
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_apply_chat_template_under_megatron_block_rejected(tmp_path):
    cfg_path = _yaml(tmp_path, [_policy(megatron={"apply_chat_template": True})])
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_data_view_keys_under_sglang_block_rejected(tmp_path):
    """Same misplacement guard as the `megatron:` case but for `sglang:`.
    Without this check the key would be quietly swallowed into the
    sglang sub-block, the Phase A guard would never see a declared
    policy override, and rollout would silently read the CLI global."""
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "sglang": {
                    "input_key": "wrong_place",
                    "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_label_key_under_sglang_block_rejected(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "sglang": {
                    "label_key": "wrong",
                    "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_apply_chat_template_under_sglang_block_rejected(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "sglang": {
                    "apply_chat_template": True,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="must be top-level policy entries"):
        parse_policy_configs(cfg_path)


def test_top_level_input_key_is_not_shadowed_by_extras(tmp_path):
    """Regression: when the .sh / megatron block contains an unrelated
    extra (e.g. num_layers), top-level input_key still ends up on
    PolicyConfig."""
    cfg_path = _yaml(
        tmp_path,
        [_policy(input_key="solver_prompt", megatron={"num_layers": 28})],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.input_key == "solver_prompt"
    assert cfg.extra_megatron_args == {"num_layers": 28}


# ── Phase A fail-fast guard (P2) ────────────────────────────────────────


def test_guard_passes_when_no_policy_declares_data_view():
    cfgs = [PolicyConfig(name="p", hf_checkpoint="/x")]
    _guard_data_views_not_yet_wired_to_rollout(cfgs)  # no raise


def test_guard_rejects_declared_input_key():
    cfgs = [PolicyConfig(name="p", hf_checkpoint="/x", input_key="solver_prompt")]
    with pytest.raises(NotImplementedError, match="not yet wired to RolloutDataSource"):
        _guard_data_views_not_yet_wired_to_rollout(cfgs)


def test_guard_rejects_declared_label_key():
    cfgs = [PolicyConfig(name="p", hf_checkpoint="/x", label_key="critic_label")]
    with pytest.raises(NotImplementedError, match="not yet wired"):
        _guard_data_views_not_yet_wired_to_rollout(cfgs)


def test_guard_rejects_declared_apply_chat_template():
    cfgs = [PolicyConfig(name="p", hf_checkpoint="/x", apply_chat_template=False)]
    with pytest.raises(NotImplementedError, match="not yet wired"):
        _guard_data_views_not_yet_wired_to_rollout(cfgs)


def test_guard_error_names_offending_policies_and_keys():
    cfgs = [
        PolicyConfig(name="solver", hf_checkpoint="/x", input_key="solver_prompt"),
        PolicyConfig(name="critic", hf_checkpoint="/x"),  # clean
        PolicyConfig(name="judge", hf_checkpoint="/x", apply_chat_template=False),
    ]
    with pytest.raises(NotImplementedError) as exc:
        _guard_data_views_not_yet_wired_to_rollout(cfgs)
    msg = str(exc.value)
    assert "solver" in msg
    assert "input_key" in msg
    assert "judge" in msg
    assert "apply_chat_template" in msg
    assert "critic" not in msg  # not an offender
