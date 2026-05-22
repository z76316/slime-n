"""Phase D tests for the `model_args_path` field, .sh parser, per-policy
defaulting/validation helpers, and `_populate_rollout_arch_fields`.

Pure-Python: no Ray, no GPUs, no Megatron. The defaulting helper is a
no-op when `slime.backends.megatron_utils.arguments` can't be imported
(test environment); the tests below assert the behavior we can observe
without that backend installed — namely the parser, the merge logic in
`parse_policy_configs`, and the helpers' Python-level contracts.

Run with:
    python -m pytest tests/test_model_args_path.py -v
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
    PolicyConfig,
    _apply_megatron_defaults,
    _load_model_sh,
    _parse_sh_model_args,
    config_to_namespace,
    parse_policy_configs,
    populate_rollout_arch_fields,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _write(tmp_path, name, body):
    p = os.path.join(tmp_path, name)
    with open(p, "w") as f:
        f.write(body)
    return p


def _yaml(tmp_path, policies):
    p = os.path.join(tmp_path, "config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"policies": policies}, f)
    return p


# ── _parse_sh_model_args ─────────────────────────────────────────────────


def test_parse_sh_kebab_to_snake_and_numeric_coercion(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        "MODEL_ARGS=(\n  --num-layers 28\n  --hidden-size 2048\n  --norm-epsilon 1e-6\n)\n",
    )
    out = _parse_sh_model_args(sh)
    assert out == {"num_layers": 28, "hidden_size": 2048, "norm_epsilon": 1e-6}


def test_parse_sh_bare_flag_and_quoted_string(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        'MODEL_ARGS=(\n  --swiglu\n  --normalization "RMSNorm"\n)\n',
    )
    assert _parse_sh_model_args(sh) == {"swiglu": True, "normalization": "RMSNorm"}


def test_parse_sh_no_flag_maps_to_store_false(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --no-rope-fusion\n)\n")
    assert _parse_sh_model_args(sh) == {"rope_fusion": False}


def test_parse_sh_no_flag_with_value_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --no-rope-fusion 1\n)\n")
    with pytest.raises(ValueError, match="store_false"):
        _parse_sh_model_args(sh)


def test_parse_sh_var_default_substituted(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        'MODEL_ARGS=(\n  --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"\n)\n',
    )
    # Even with the env var set, parser uses the literal default.
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "9999999"
    try:
        assert _parse_sh_model_args(sh) == {"rotary_base": 1000000}
    finally:
        del os.environ["MODEL_ARGS_ROTARY_BASE"]


def test_parse_sh_var_without_default_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", 'MODEL_ARGS=(\n  --rotary-base "${MODEL_ARGS_ROTARY_BASE}"\n)\n')
    with pytest.raises(ValueError, match="unsupported bash interpolation"):
        _parse_sh_model_args(sh)


def test_parse_sh_command_substitution_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --num-layers $(echo 28)\n)\n")
    with pytest.raises(ValueError, match="unsupported bash interpolation"):
        _parse_sh_model_args(sh)


def test_parse_sh_bare_shell_var_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --num-layers $NLAYERS\n)\n")
    with pytest.raises(ValueError, match="unsupported bash interpolation"):
        _parse_sh_model_args(sh)


def test_parse_sh_no_model_args_array_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", "echo nothing\n")
    with pytest.raises(ValueError, match="no MODEL_ARGS"):
        _parse_sh_model_args(sh)


def test_parse_sh_unsupported_error_uses_supported_path_language(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --num-layers $NLAYERS\n)\n")
    with pytest.raises(ValueError) as exc:
        _parse_sh_model_args(sh)
    msg = str(exc.value)
    assert "unsupported bash interpolation" in msg
    assert "Use one of these supported paths" in msg
    assert "Workaround" not in msg


def test_parse_sh_comments_and_blank_lines_skipped(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        "MODEL_ARGS=(\n  # comment\n\n  --num-layers 4  # inline\n)\n",
    )
    assert _parse_sh_model_args(sh) == {"num_layers": 4}


def test_parse_sh_multi_value_on_scalar_flag_raises(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --num-layers 28 29\n)\n")
    with pytest.raises(ValueError, match="multi-value allowlist"):
        _parse_sh_model_args(sh)


def test_parse_sh_multi_value_allowed_for_spec(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --spec mod fn\n)\n")
    assert _parse_sh_model_args(sh) == {"spec": ["mod", "fn"]}


# ── _load_model_sh ───────────────────────────────────────────────────────


def test_load_model_sh_absolute(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --num-layers 7\n)\n")
    assert _load_model_sh(sh) == {"num_layers": 7}


def test_load_model_sh_relative_resolves_against_repo_root():
    # The shipped Qwen3-0.6B model file exists in the repo.
    out = _load_model_sh("scripts/models/qwen3-0.6B.sh")
    assert out["num_layers"] == 28
    assert out["hidden_size"] == 1024


def test_load_model_sh_missing_raises():
    with pytest.raises(FileNotFoundError):
        _load_model_sh("scripts/models/__definitely_not_a_model__.sh")


def test_load_model_sh_relative_parse_error_uses_original_path(tmp_path, monkeypatch):
    from slime.utils import policy_config

    model_dir = tmp_path / "scripts" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "wrapper.sh").write_text("echo no model args\n")

    monkeypatch.setattr(policy_config, "_repo_root", lambda: str(tmp_path))
    with pytest.raises(ValueError) as exc:
        policy_config._load_model_sh("scripts/models/wrapper.sh")
    msg = str(exc.value)
    assert msg.startswith("scripts/models/wrapper.sh:")
    assert str(tmp_path) not in msg
    assert "Use one of these supported paths" in msg


# ── parse_policy_configs merge behavior ──────────────────────────────────


def test_model_args_path_loads_into_extras_and_known(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        "MODEL_ARGS=(\n  --num-layers 28\n  --sequence-parallel\n)\n",
    )
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"model_args_path": sh, "tensor_model_parallel_size": 2},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )
    cfgs = parse_policy_configs(cfg_path)
    assert len(cfgs) == 1
    cfg = cfgs[0]
    # --sequence-parallel is a declared PolicyConfig field; landed in known bucket.
    assert cfg.sequence_parallel is True
    # --num-layers is an unknown PolicyConfig key; landed in extras.
    assert cfg.extra_megatron_args is not None
    assert cfg.extra_megatron_args["num_layers"] == 28


def test_norm_epsilon_from_model_args_path_mirrors_layernorm_epsilon(tmp_path):
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --norm-epsilon 1e-6\n)\n")
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"model_args_path": sh},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )

    cfg = parse_policy_configs(cfg_path)[0]

    assert cfg.extra_megatron_args is not None
    assert cfg.extra_megatron_args["norm_epsilon"] == 1e-6
    assert cfg.extra_megatron_args["layernorm_epsilon"] == 1e-6


def test_inline_layernorm_epsilon_mirrors_norm_epsilon(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"layernorm_epsilon": 1e-6},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )

    cfg = parse_policy_configs(cfg_path)[0]

    assert cfg.extra_megatron_args is not None
    assert cfg.extra_megatron_args["norm_epsilon"] == 1e-6
    assert cfg.extra_megatron_args["layernorm_epsilon"] == 1e-6


def test_explicit_norm_epsilon_pair_is_preserved(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"norm_epsilon": 1e-6, "layernorm_epsilon": 1e-5},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )

    cfg = parse_policy_configs(cfg_path)[0]

    assert cfg.extra_megatron_args is not None
    assert cfg.extra_megatron_args["norm_epsilon"] == 1e-6
    assert cfg.extra_megatron_args["layernorm_epsilon"] == 1e-5


def test_inline_megatron_beats_sh_value(tmp_path):
    sh = _write(
        tmp_path,
        "m.sh",
        "MODEL_ARGS=(\n  --num-layers 28\n  --sequence-parallel\n)\n",
    )
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {
                    "model_args_path": sh,
                    "sequence_parallel": False,  # inline override of a known field
                    "num_layers": 99,  # inline override of an extra
                },
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.sequence_parallel is False
    assert cfg.extra_megatron_args["num_layers"] == 99


def test_model_args_path_none_is_noop(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"num_layers": 12},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.model_args_path is None
    assert cfg.extra_megatron_args == {"num_layers": 12}


def test_model_args_path_missing_file_raises(tmp_path):
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"model_args_path": "/tmp/__does_not_exist__.sh"},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        parse_policy_configs(cfg_path)


def test_model_args_path_overlap_with_policyconfig_field(tmp_path):
    """An upstream .sh with --sequence-parallel (a PolicyConfig field)
    must NOT raise — it gets routed into megatron_known via the split
    path, identical to the user typing `sequence_parallel: true` inline.
    """
    sh = _write(tmp_path, "m.sh", "MODEL_ARGS=(\n  --sequence-parallel\n)\n")
    cfg_path = _yaml(
        tmp_path,
        [
            {
                "name": "p",
                "hf_checkpoint": "/x",
                "num_gpus_per_node": 1,
                "megatron": {"model_args_path": sh},
                "sglang": {"server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
            }
        ],
    )
    cfg = parse_policy_configs(cfg_path)[0]
    assert cfg.sequence_parallel is True


# ── _apply_megatron_defaults ─────────────────────────────────────────────


def test_apply_megatron_defaults_clears_stale_padded_vocab():
    ns = argparse.Namespace(padded_vocab_size=8888, vocab_size=151936)
    cfg = PolicyConfig(name="p", hf_checkpoint="/x", extra_megatron_args={"vocab_size": 151936})
    _apply_megatron_defaults(ns, cfg)
    # Megatron isn't installed in this env, so set_default_megatron_args is a no-op;
    # the helper still cleared the stale derived field.
    assert ns.padded_vocab_size is None


def test_apply_megatron_defaults_respects_explicit_padded_vocab():
    ns = argparse.Namespace(padded_vocab_size=8888, vocab_size=151936)
    cfg = PolicyConfig(
        name="p",
        hf_checkpoint="/x",
        extra_megatron_args={"vocab_size": 151936, "padded_vocab_size": 152064},
    )
    _apply_megatron_defaults(ns, cfg)
    # Explicit value in extras wins (re-apply step after defaulting).
    assert ns.padded_vocab_size == 152064


def test_apply_megatron_defaults_reapplies_extras_after_defaults():
    """`set_default_megatron_args` has unconditional writes (e.g.
    max_position_embeddings = seq_length). The re-apply loop must put
    user-explicit extras back."""
    ns = argparse.Namespace()
    cfg = PolicyConfig(name="p", hf_checkpoint="/x", extra_megatron_args={"custom_flag": "user_value"})
    _apply_megatron_defaults(ns, cfg)
    assert ns.custom_flag == "user_value"


# ── config_to_namespace + extras ─────────────────────────────────────────


def test_config_to_namespace_calls_apply_megatron_defaults():
    """Integration: config_to_namespace must invoke _apply_megatron_defaults
    as its last step. We observe this through the same stale-padded_vocab
    behavior as the unit test."""
    base = argparse.Namespace(
        hf_checkpoint=None,
        padded_vocab_size=8888,
        tokenizer_model=None,
        tokenizer_type=None,
        eps_clip_high=None,
        n_samples_per_prompt=4,
        use_dynamic_batch_size=False,
        log_probs_max_tokens_per_gpu=None,
        max_tokens_per_gpu=None,
        ref_ckpt_step=None,
    )
    cfg = PolicyConfig(
        name="p",
        hf_checkpoint="/x",
        megatron_to_hf_mode="bridge",
        extra_megatron_args={"vocab_size": 151936},
    )
    ns = config_to_namespace(cfg, base)
    assert ns.padded_vocab_size is None  # stale cleared via _apply_megatron_defaults


# ── _populate_rollout_arch_fields ────────────────────────────────────────

# Imported lazily to avoid pulling in slime.backends.* at collection time.


def _engine_cfg(name="p", num_gpus=1):
    return PolicyConfig(
        name=name,
        hf_checkpoint="/x",
        sglang_num_nodes=1,
        megatron_num_nodes=1,
        num_gpus_per_node=num_gpus,
    )


def _standalone_cfg(name="critic"):
    return PolicyConfig(
        name=name,
        hf_checkpoint="/x",
        sglang_num_nodes=0,
        megatron_num_nodes=1,
        num_gpus_per_node=1,
        advantage_estimator="ppo",
    )


def test_populate_rollout_arch_homogeneous():
    base = argparse.Namespace(num_layers=None, use_rollout_routing_replay=False)
    cfgs = [_engine_cfg("a"), _engine_cfg("b")]
    policy_args = [argparse.Namespace(num_layers=28), argparse.Namespace(num_layers=28)]
    populate_rollout_arch_fields(base, cfgs, policy_args)
    assert base.num_layers == 28


def test_populate_rollout_arch_mixed_with_routing_replay_raises():
    base = argparse.Namespace(num_layers=None, use_rollout_routing_replay=True)
    cfgs = [_engine_cfg("a"), _engine_cfg("b")]
    policy_args = [argparse.Namespace(num_layers=28), argparse.Namespace(num_layers=64)]
    with pytest.raises(ValueError, match="mixed-arch"):
        populate_rollout_arch_fields(base, cfgs, policy_args)


def test_populate_rollout_arch_missing_num_layers_with_routing_replay_raises():
    base = argparse.Namespace(num_layers=None, use_rollout_routing_replay=True)
    cfgs = [_engine_cfg("actor"), _engine_cfg("teacher")]
    policy_args = [argparse.Namespace(num_layers=28), argparse.Namespace(num_layers=None)]
    with pytest.raises(ValueError) as exc:
        populate_rollout_arch_fields(base, cfgs, policy_args)
    msg = str(exc.value)
    assert "requires num_layers" in msg
    assert "teacher" in msg


def test_populate_rollout_arch_ignores_standalone_policies():
    """A trainable-standalone policy (e.g. PPO critic) has a different
    num_layers but should not be considered — it never serves rollout."""
    base = argparse.Namespace(num_layers=None, use_rollout_routing_replay=False)
    cfgs = [_engine_cfg("actor"), _standalone_cfg("critic")]
    policy_args = [argparse.Namespace(num_layers=28), argparse.Namespace(num_layers=48)]
    populate_rollout_arch_fields(base, cfgs, policy_args)
    assert base.num_layers == 28


def test_populate_rollout_arch_mixed_without_routing_replay_does_not_raise():
    """When routing replay is off, mixed-arch is fine: rollout doesn't
    reshape routed experts. base_args.num_layers is left unset (no single
    value to copy)."""
    base = argparse.Namespace(num_layers=None, use_rollout_routing_replay=False)
    cfgs = [_engine_cfg("a"), _engine_cfg("b")]
    policy_args = [argparse.Namespace(num_layers=28), argparse.Namespace(num_layers=64)]
    populate_rollout_arch_fields(base, cfgs, policy_args)
    # No single value, no raise; base_args.num_layers untouched.
    assert base.num_layers is None


# ── parse_args(skip_megatron_model_validation=...) ───────────────────────


def test_parse_args_kwarg_in_source():
    """The kwarg is wired in. Source-level grep avoids importing
    slime.utils.arguments (which pulls sglang_router at import time and
    isn't available in this test env)."""
    arguments_py = os.path.join(_REPO_ROOT, "slime/utils/arguments.py")
    with open(arguments_py) as f:
        src = f.read()
    assert "skip_megatron_model_validation: bool = False" in src
    assert "skip_hf_validate=pre.debug_rollout_only or skip_megatron_model_validation" in src
    assert "not skip_megatron_model_validation" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
