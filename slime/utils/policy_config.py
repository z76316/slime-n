"""PolicyConfig dataclass + flat-fields parser for multi-policy training.

Parses examples/<run>/config.yaml of the form:

    policies:
      - name: solver
        role: actor
        hf_checkpoint: ...
        load: ...
        save: ...
        ref_load: ...                   # optional
        buffer_mode: split | shared
        num_gpus_per_node: 8
        megatron_num_nodes: 1
        sglang_num_nodes: 1
        megatron:                       # nested: training fields
          tensor_model_parallel_size: ...
          ...
        sglang:                         # nested: engine deployment fields
          model_path: ...               # defaults to top-level hf_checkpoint
          server_groups: [...]
          ...

Parser flattens megatron[*] into top-level PolicyConfig fields, keeps sglang as
a raw dict (later projected to a SglangConfig.ModelConfig). 1:1 pairing is
implicit: PolicyConfig.sglang_server defaults to PolicyConfig.name.

Diverges from upstream parse_megatron_role_args (commit f65c6e8): no overrides:
sub-dict, no silent critic field forcing. Validation only.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Flags whose argparse nargs is >1 in upstream Megatron. Anything not in
# this allowlist is treated as scalar; passing >1 value to a scalar flag
# is a user typo (e.g. `--num-layers 28 29`) and raises.
_MULTI_VALUE_FLAGS: frozenset[str] = frozenset({"spec"})


def _repo_root() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_model_args_tokens(path: str, display_path: str | None = None) -> list[str]:
    """Extract MODEL_ARGS=(...) body from a .sh file as a token list.

    Substitutes `${VAR:-default}` → literal default (env IGNORED). Rejects
    any other bash interpolation (`${VAR}`, `$(cmd)`, bare `$VAR`,
    arithmetic, etc.) with a clear error pointing at the offending line.
    """
    import re
    import shlex

    label = display_path or path
    with open(path) as f:
        text = f.read()
    m = re.search(r"MODEL_ARGS=\(\s*(.*?)\s*\)", text, re.DOTALL)
    if not m:
        raise ValueError(
            f"{label}: no MODEL_ARGS=( ... ) array found. "
            "Use one of these supported paths: keep this script on the legacy "
            "`source ...` CLI path, or declare the architecture fully inline "
            "in the policy's `megatron:` block."
        )
    body = m.group(1)

    _interp_default = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}")
    _bare_shell_var = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
    cleaned: list[str] = []
    for ln in body.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        ln = _interp_default.sub(r"\1", ln)
        if "${" in ln or "$(" in ln or _bare_shell_var.search(ln):
            raise ValueError(
                f"{label}: unsupported bash interpolation in `{ln}`; "
                "only ${VAR:-default} is supported. Use one of these supported "
                "paths: keep this script on the legacy `source ...` CLI path, "
                "or declare the affected architecture fields inline in the "
                "policy's `megatron:` block."
            )
        cleaned.append(ln)
    return shlex.split(" ".join(cleaned))


def _parse_sh_model_args(path: str, display_path: str | None = None) -> dict:
    """Parse a MODEL_ARGS=( ... ) bash array into a snake_case-keyed dict.

    See plan_model_field.md "Bash → dict translation rules" for the full
    contract. Coerces numbers (int, float), keeps strings otherwise.
    `--no-foo` maps to `foo: False` (Megatron store_false convention).
    """
    label = display_path or path
    tokens = _extract_model_args_tokens(path, display_path=label)

    def _coerce(raw: str):
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
        return raw

    result: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"{label}: unexpected token {tok!r} at MODEL_ARGS[{i}]")
        is_store_false = tok.startswith("--no-")
        raw_name = tok[5:] if is_store_false else tok[2:]
        key = raw_name.replace("-", "_")
        j = i + 1
        while j < len(tokens) and not tokens[j].startswith("--"):
            j += 1
        values = [_coerce(v) for v in tokens[i + 1 : j]]
        if not values:
            result[key] = False if is_store_false else True
        elif is_store_false:
            raise ValueError(
                f"{label}: store_false flag {tok!r} got a value {values!r}; " "`--no-X` flags don't take arguments."
            )
        elif len(values) == 1:
            result[key] = values[0]
        else:
            if key not in _MULTI_VALUE_FLAGS:
                raise ValueError(
                    f"{label}: flag {tok!r} got {len(values)} values "
                    f"{values!r} but is not in the multi-value allowlist "
                    f"({sorted(_MULTI_VALUE_FLAGS)})."
                )
            result[key] = values
        i = j
    return result


# Megatron decouples some CLI flag names from the dataclass field they store
# into via `field(metadata={"argparse_meta": {"arg_names": [...]}})` in
# megatron/core/transformer/transformer_config.py. Our `.sh` parser does
# naive kebab→snake on the flag name (e.g. `--norm-epsilon` →
# `norm_epsilon`), but Megatron's argparse would store into the real field
# (`layernorm_epsilon`). When MODEL_ARGS flows through this parser instead
# of through Megatron's CLI, the renamed attribute is never populated.
#
# Each entry is (naive_dest, real_field, invert):
#   - invert=False: store_true / scalar flags whose dest name was renamed.
#                   Both keys end up equal.
#   - invert=True : "negative" toggle flags (`--disable-FOO`, `--no-FOO`)
#                   whose presence on the CLI sets the underlying field
#                   to the inverse value.
#
# Audit source: upstream megatron/core/transformer/transformer_config.py.
# Grow this table as new Megatron releases add metadata-renamed flags.
_MEGATRON_FLAG_RENAMES: tuple[tuple[str, str, bool], ...] = (
    ("norm_epsilon", "layernorm_epsilon", False),
    ("disable_bias_linear", "add_bias_linear", True),
    ("apply_layernorm_1p", "layernorm_zero_centered_gamma", False),
    ("disable_mamba_mem_eff_path", "use_mamba_mem_eff_path", True),
    ("fp8_format", "fp8", False),
    ("fp4_format", "fp4", False),
    ("fp4_param_gather", "fp4_param", False),
)


def _reconcile_megatron_flag_renames(extras: dict[str, Any]) -> dict[str, Any]:
    """Mirror flag-name / field-name pairs that Megatron's argparse renames.

    When only one of a known pair is provided in `extras`, copy the value
    onto the other (inverting booleans for `--disable-FOO`-style toggles)
    so the downstream Megatron-side field stays populated. When both are
    provided, respect both — the user has explicitly disagreed and we
    don't pretend to know which they meant.

    See `_MEGATRON_FLAG_RENAMES` for the table.
    """
    out = dict(extras)
    for naive, real, invert in _MEGATRON_FLAG_RENAMES:
        n_in = naive in out
        r_in = real in out
        if n_in and not r_in:
            out[real] = (not out[naive]) if invert else out[naive]
        elif r_in and not n_in:
            out[naive] = (not out[real]) if invert else out[real]
    return out


def _apply_megatron_defaults(policy_args, cfg) -> None:
    """Re-derive Megatron defaults on a per-policy namespace.

    `set_default_megatron_args` only fills `padded_vocab_size` when it's
    currently None — so if `base_args` already had a `padded_vocab_size`
    (e.g. from a legacy sourced MODEL_ARGS) and the policy overrides
    `vocab_size`, the stale value would survive. Clear it first.

    `set_default_megatron_args` also has unconditional writes (e.g.
    `args.max_position_embeddings = args.seq_length`) that would clobber
    user-explicit values in extras. Re-apply extras AFTER defaulting so
    user-explicit values always win.
    """
    try:
        from slime.backends.megatron_utils.arguments import set_default_megatron_args
    except ModuleNotFoundError:
        # Megatron isn't installed (tests run on the slime-n config plumbing
        # without the full backend). The defaulting helper is a no-op in
        # that environment — extras still get applied directly.
        set_default_megatron_args = None

    extras = cfg.extra_megatron_args or {}
    vocab_inputs_overridden = "vocab_size" in extras or "make_vocab_size_divisible_by" in extras
    if vocab_inputs_overridden and "padded_vocab_size" not in extras:
        policy_args.padded_vocab_size = None
    if set_default_megatron_args is not None:
        set_default_megatron_args(policy_args)
    for k, v in extras.items():
        setattr(policy_args, k, v)


def _validate_hf_per_policy(policy_args) -> None:
    """Re-run HF + allgather-CP validators against the policy's hf_checkpoint.

    Mirrors what `megatron_parse_args` does at parse time, but per-policy
    so each Megatron actor's arch is checked against its own HF config.
    Skips when `hf_checkpoint` is empty.
    """
    if not getattr(policy_args, "hf_checkpoint", None):
        return
    from transformers import AutoConfig

    from slime.backends.megatron_utils.arguments import _hf_validate_args, _validate_allgather_cp_supported

    hf_config = AutoConfig.from_pretrained(policy_args.hf_checkpoint, trust_remote_code=True)
    _hf_validate_args(policy_args, hf_config)
    _validate_allgather_cp_supported(policy_args, hf_config)


def populate_rollout_arch_fields(base_args, policy_configs, all_policy_args) -> None:
    """Populate Megatron arch fields read by the rollout code path.

    `sglang_rollout.py` reads `args.num_layers` for routed-expert reshape
    (gated on `use_rollout_routing_replay`). After the multi-policy
    parse-flow refactor, `base_args.num_layers` is None until something
    sets it. Compute the set of `num_layers` values across
    engine-hosting policies only — standalone-Megatron policies (e.g.
    PPO critic) never produce rollout tokens.

    Raises if `use_rollout_routing_replay` is True and engine-hosting
    policies are missing `num_layers` or disagree on `num_layers`.
    """
    engine_arch_by_name = {
        cfg.name: getattr(pa, "num_layers", None)
        for cfg, pa in zip(policy_configs, all_policy_args, strict=True)
        if has_sglang_engine(cfg)
    }
    engine_archs = {num_layers for num_layers in engine_arch_by_name.values() if num_layers is not None}
    if getattr(base_args, "use_rollout_routing_replay", False):
        missing = sorted(name for name, num_layers in engine_arch_by_name.items() if num_layers is None)
        if missing:
            raise ValueError(
                "routed-expert rollout requires num_layers for every engine-hosting policy; "
                f"missing for policies: {missing}. "
                "Set megatron.model_args_path to a supported static MODEL_ARGS script, "
                "or declare num_layers inline in each policy's megatron block."
            )
        if len(engine_archs) > 1:
            raise ValueError(
                "mixed-arch + routed-expert rollout not supported in v1: "
                f"engine-hosting policies disagree on num_layers ({sorted(engine_archs)}). "
                "Either disable use_rollout_routing_replay or align num_layers across "
                "engine-hosting policies."
            )
    if len(engine_archs) == 1:
        base_args.num_layers = next(iter(engine_archs))


def _load_model_sh(path: str) -> dict:
    """Resolve `path` (relative to repo root or absolute), then parse it."""
    import os

    sh_path = path if os.path.isabs(path) else os.path.join(_repo_root(), path)
    if not os.path.exists(sh_path):
        raise FileNotFoundError(f"model_args_path: {sh_path!r} not found (from {path!r}).")
    return _parse_sh_model_args(sh_path, display_path=path)


@dataclasses.dataclass
class PolicyConfig:
    # ── identity ──
    name: str
    role: str = "actor"  # only "actor" supported in v1; critic deferred

    # ── model / checkpoint ──
    hf_checkpoint: str = ""
    load: str | None = None
    save: str | None = None
    ref_load: str | None = None  # KL ref; with_ref=True when kl_coef != 0 or use_kl_loss=True

    # ── Megatron parallel ──
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Megatron numerical / dropout (RL-correctness defaults) ──
    # Megatron defaults attention_dropout / hidden_dropout to 0.1; RL training
    # needs deterministic forward (rollout-time and train-time log probs must
    # match) so we default these to 0 for any policy.
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True

    # ── Memory: chunked log-prob computation ──
    # >0 chunks the [T, V] logits tensor along T to reduce forward peak; -1
    # disables chunking. Critical for vocab-heavy models on tight VRAM (e.g.
    # Qwen3 with 152K vocab on L40S).
    log_probs_chunk_size: int = -1

    # ── Recompute (memory/compute trade) ──
    recompute_granularity: str | None = None  # "full" | "selective" | None
    recompute_method: str | None = None  # "uniform" | "block" | None
    recompute_num_layers: int | None = None

    # ── Weight-load mode ──
    # "raw" (default): expects a Megatron torch_dist checkpoint at `load`.
    # "bridge": when `load` is unset or stale, slime falls back to loading from
    # `hf_checkpoint` via mbridge. Useful for first-time runs from an HF model.
    megatron_to_hf_mode: str = "raw"

    # ── Batching ──
    micro_batch_size: int = 1
    global_batch_size: int = 64
    use_dynamic_batch_size: bool = False
    max_tokens_per_gpu: int | None = None
    log_probs_max_tokens_per_gpu: int | None = None  # falls back to max_tokens_per_gpu

    # ── Optimizer (per-policy — each actor has its own optimizer state) ──
    optimizer: str = "adam"
    lr: float = 1.0e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    optimizer_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    use_precision_aware_optimizer: bool = False

    # ── Loss / RL hyperparameters ──
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    kl_coef: float = 0.0
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    use_kl_loss: bool = False
    custom_advantage_function_path: str | None = None
    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 1
    rewards_normalization: bool = True
    grpo_std_normalization: bool = True

    # ── Multi-policy orchestration ──
    sglang_server: str | None = None  # 1:1 pairing; defaults to PolicyConfig.name
    buffer_mode: str = "split"  # "split" | "shared"
    # trainable=False marks a frozen producer (e.g., OPD Megatron teacher):
    # the train loop runs its forward-only train() and merges the returned
    # dict into trainable consumers' external_data. Frozen actors skip
    # weights_backuper / weight_updater / optimizer-state save / weight push.
    trainable: bool = True

    # ── On-policy distillation (per-policy; legacy was CLI-global) ──
    # use_opd / opd_type / opd_kl_coef live on the trainable consumer (the
    # student). For the new external-data path, set opd_type="megatron_actor"
    # — this skips the legacy in-process tag teacher path at
    # placement_group.py:159 (which fires only when opd_type == "megatron").
    # opd_teacher_load is kept for the legacy in-process path only.
    use_opd: bool = False
    opd_type: str | None = None
    opd_kl_coef: float = 1.0
    opd_teacher_load: str | None = None

    # ── GPU placement (cluster-level num_gpus_per_node, per-side node counts) ──
    num_gpus_per_node: int = 8
    megatron_num_nodes: int = 1
    sglang_num_nodes: int = 1

    # ── sglang sub-block (raw dict, projected to SglangConfig.ModelConfig later) ──
    sglang: dict | None = None

    # ── per-policy Megatron flag overrides (architecture, etc.) ──
    # Catches any key in the YAML `megatron:` block that is not a declared
    # PolicyConfig field, plus any extras loaded from `model_args_path`.
    # config_to_namespace applies each as setattr on the per-policy ns, so a
    # YAML key like `num_layers: 48` (or one loaded from a .sh) overrides
    # whatever came from the global CLI MODEL_ARGS. Needed when N policies
    # in one run don't share architecture.
    extra_megatron_args: dict | None = None

    # ── model args file reference ──
    # Path to an upstream `scripts/models/<name>.sh` (the bash
    # `MODEL_ARGS=(...)` array). When set, the parser reads the .sh,
    # converts kebab→snake, and merges parsed flags UNDER any inline
    # `megatron:` block values (inline wins). Path is resolved relative to
    # the slime-n repo root; absolute paths also accepted. Lets multiple
    # policies reference the same arch without bash sourcing in the run
    # script. Note: only `${VAR:-default}` interpolation is supported
    # (literal default is used; env var is IGNORED); `${VAR}`, `$(cmd)`,
    # and bare `$VAR` are rejected.
    model_args_path: str | None = None


def has_sglang_engine(cfg: PolicyConfig) -> bool:
    """True iff the policy hosts an SGLang engine.

    `cfg.sglang_server` is NOT a reliable predicate — the parser sets it to
    `entry["name"]` unconditionally, even when `sglang_num_nodes == 0`.
    `sglang_num_nodes` is the source of truth for engine existence.
    """
    return cfg.sglang_num_nodes > 0


def is_critic_shape(cfg: PolicyConfig) -> bool:
    """PPO critic shape: trainable Megatron-only policy with ppo advantage_estimator.

    Used by placement_group.py to wire role="critic" (which makes the model
    provider attach a 1-dim value head and skips weight_updater init) and to
    flip args.use_critic=True on sibling actors. NOT used by the multi-policy
    driver — the driver partitions by `has_sglang_engine` (shape), not by
    PPO-specific role semantics.
    """
    return cfg.trainable and not has_sglang_engine(cfg) and cfg.advantage_estimator == "ppo"


def validate_policy_config(cfg: PolicyConfig) -> None:
    if cfg.role not in {"actor", "critic"}:
        raise ValueError(f"{cfg.name}: role must be 'actor' or 'critic', got {cfg.role!r}")
    if cfg.sglang_server is None:
        raise ValueError(f"{cfg.name}: policy requires sglang_server (1:1 pairing; defaults to policy name)")
    if cfg.buffer_mode not in {"shared", "split"}:
        raise ValueError(f"{cfg.name}: buffer_mode must be 'shared' or 'split'")
    if not cfg.hf_checkpoint:
        raise ValueError(f"{cfg.name}: hf_checkpoint required")

    # Sglang placement consistency. Frozen producers (e.g. OPD Megatron
    # teacher) don't run an engine; their sglang sub-block / sglang_num_nodes
    # are unused, so don't apply the placement check.
    if cfg.sglang is not None and cfg.trainable:
        sglang_total = cfg.sglang_num_nodes * cfg.num_gpus_per_node
        groups_total = sum(g["num_gpus"] for g in cfg.sglang.get("server_groups", []))
        if sglang_total != groups_total:
            raise ValueError(
                f"{cfg.name}: sglang_num_nodes × num_gpus_per_node ({sglang_total}) "
                f"must equal sum of sglang.server_groups[].num_gpus ({groups_total})"
            )


def parse_policy_configs(config_path: str) -> list[PolicyConfig]:
    """Parse config.yaml into list[PolicyConfig].

    Each policies[i] entry has top-level fields and two nested sub-blocks (megatron, sglang).
    Parser flattens megatron[*] into PolicyConfig fields; sglang stays as a raw dict
    (build_sglang_config_from_policies projects it to a SglangConfig later).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if "policies" not in raw or not isinstance(raw["policies"], list):
        raise ValueError("config must have top-level 'policies' list")

    configs: list[PolicyConfig] = []
    for entry in raw["policies"]:
        # sglang.model_path defaults to top-level hf_checkpoint
        sglang_block = dict(entry.get("sglang") or {})
        sglang_block.setdefault("model_path", entry.get("hf_checkpoint"))

        # Split the megatron sub-block into (a) keys that match PolicyConfig
        # dataclass fields (declared) and (b) extras (anything else). Extras
        # are passed through as per-policy CLI overrides via
        # extra_megatron_args (architecture flags like num_layers / num_experts
        # live there when policies don't share architecture).
        known_field_names = {f.name for f in dataclasses.fields(PolicyConfig)}
        megatron_block = entry.get("megatron") or {}
        megatron_known = {k: v for k, v in megatron_block.items() if k in known_field_names}
        megatron_extras = {k: v for k, v in megatron_block.items() if k not in known_field_names}

        # When megatron.model_args_path is set, load the .sh and merge its
        # parsed flags UNDER the inline `megatron:` block. Real upstream
        # files contain legitimate PolicyConfig fields (e.g.
        # --sequence-parallel in gpt-oss-20B.sh), so split into the same
        # two buckets as the inline block; inline always wins on conflict.
        model_args_path = megatron_known.get("model_args_path")
        if model_args_path:
            parsed = _load_model_sh(model_args_path)
            parsed_known = {k: v for k, v in parsed.items() if k in known_field_names}
            parsed_extras = {k: v for k, v in parsed.items() if k not in known_field_names}
            megatron_known = {**parsed_known, **megatron_known}
            megatron_extras = {**parsed_extras, **megatron_extras}
        megatron_extras = _reconcile_megatron_flag_renames(megatron_extras)

        flat: dict[str, Any] = {
            "name": entry["name"],
            "role": entry.get("role", "actor"),
            "trainable": entry.get("trainable", True),
            "hf_checkpoint": entry.get("hf_checkpoint", ""),
            "load": entry.get("load"),
            "save": entry.get("save"),
            "ref_load": entry.get("ref_load"),
            "buffer_mode": entry.get("buffer_mode", "split"),
            "num_gpus_per_node": entry.get("num_gpus_per_node", 8),
            "megatron_num_nodes": entry.get("megatron_num_nodes", 1),
            "sglang_num_nodes": entry.get("sglang_num_nodes", 1),
            **megatron_known,
            "extra_megatron_args": megatron_extras or None,
            "sglang": sglang_block,
            "sglang_server": entry["name"],  # 1:1 pairing: server name = policy name
        }
        configs.append(PolicyConfig(**flat))

    for cfg in configs:
        validate_policy_config(cfg)
    _validate_unique_names(configs)
    _validate_unique_sglang_servers(configs)
    _validate_shared_buffer_consistency(configs)
    return configs


def _validate_unique_names(configs: list[PolicyConfig]) -> None:
    names = [c.name for c in configs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate policy names: {names}")


def _validate_unique_sglang_servers(configs: list[PolicyConfig]) -> None:
    """1:1 pairing: each engine has exactly one trainable owner."""
    servers = [c.sglang_server for c in configs]
    if len(servers) != len(set(servers)):
        raise ValueError(f"two policies cannot push to the same sglang_server: {servers}")


def _validate_shared_buffer_consistency(configs: list[PolicyConfig]) -> None:
    """Shared-buffer policies must agree on advantage_estimator and n_samples_per_prompt
    (group-norm depends on these)."""
    shared = [c for c in configs if c.buffer_mode == "shared"]
    if len(shared) <= 1:
        return
    estimators = {c.advantage_estimator for c in shared}
    if len(estimators) > 1:
        raise ValueError(
            f"shared-buffer policies must agree on advantage_estimator: "
            f"{ {c.name: c.advantage_estimator for c in shared} }"
        )
    nsp = {c.n_samples_per_prompt for c in shared}
    if len(nsp) > 1:
        raise ValueError(
            f"shared-buffer policies must agree on n_samples_per_prompt: "
            f"{ {c.name: c.n_samples_per_prompt for c in shared} }"
        )


def build_sglang_config_from_policies(configs: list[PolicyConfig]):
    """Project each policy's sglang sub-block into a SglangConfig.ModelConfig.

    The user-facing sglang sub-block is FLAT (no overrides: indirection).
    Model-level fields (model_path, num_gpus_per_engine, update_weights, server_groups)
    map directly onto ModelConfig. All other fields are sglang ServerArgs and get
    folded into each ServerGroupConfig.overrides for upstream's _compute_server_args.
    """
    from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

    MODEL_FIELDS = {"model_path", "num_gpus_per_engine", "update_weights", "server_groups"}

    models = []
    for cfg in configs:
        # Skip policies that don't host an engine. Predicate is per-policy GPU
        # allocation, not trainability — frozen standalone SGLang engines
        # (m✗ s✓ trainable=false, e.g. judge / RM / OPD SGLang teacher) need a
        # ModelConfig too. Megatron-only frozen producers (m✓ s✗) have
        # sglang_num_nodes=0 and are correctly skipped.
        if cfg.sglang_num_nodes == 0:
            continue
        if cfg.sglang is None:
            raise ValueError(f"{cfg.name}: missing 'sglang' sub-block in config")
        sg = dict(cfg.sglang)

        # Server-args (everything except model-level fields) flow into each group's overrides.
        server_arg_overrides = {k: v for k, v in sg.items() if k not in MODEL_FIELDS}

        groups = []
        for g in sg.get("server_groups", []):
            g = dict(g)
            # Per-group overrides win over model-level server-args.
            merged_overrides = {**server_arg_overrides, **g.pop("overrides", {})}
            groups.append(ServerGroupConfig(**g, overrides=merged_overrides))

        models.append(
            ModelConfig(
                name=cfg.name,  # 1:1 pairing: server name = policy name
                model_path=sg.get("model_path"),
                num_gpus_per_engine=sg.get("num_gpus_per_engine"),
                update_weights=sg.get("update_weights", True),
                server_groups=groups,
            )
        )
    return SglangConfig(models=models)


def derive_cluster_sizing(configs: list[PolicyConfig], colocate: bool) -> tuple[int, int, int]:
    """Compute (actor_gpus, rollout_gpus, total_gpus) from PolicyConfig list.

    Used by:
      - launcher (to set ray start --num-gpus)
      - create_placement_groups_multi (to size the global PG)
    """
    actor_gpus = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in configs)
    # Sum across all policies — Megatron-only producers (m✓ s✗) contribute 0
    # via their sglang_num_nodes=0; engine-hosting policies (trainable paired
    # and frozen standalone) contribute their actual engine GPUs.
    rollout_gpus = sum(c.sglang_num_nodes * c.num_gpus_per_node for c in configs)
    total_gpus = max(actor_gpus, rollout_gpus) if colocate else actor_gpus + rollout_gpus
    return actor_gpus, rollout_gpus, total_gpus


@dataclasses.dataclass
class PolicyHandle:
    """One trainable Megatron actor + its 1:1-paired sglang engine handle.

    Built by create_training_models_multi (slime.ray.placement_group). The driver
    iterates a dict[name, PolicyHandle] returned from there.
    """

    config: PolicyConfig
    args: Any  # PolicyConfig projected onto a Namespace for downstream Megatron code
    train_group: Any  # RayTrainGroup
    # Runtime-derived role passed to allocate_train_group and async_init.
    # For shape-derived critic, this is "critic"; otherwise mirrors cfg.role.
    # Kept distinct from cfg.role so the YAML-declared value stays intact for
    # logs/saves.
    role_eff: str = "actor"


def config_to_namespace(cfg: PolicyConfig, base_args):
    """Project PolicyConfig fields onto a Namespace, copying everything directly.

    Pulls non-policy globals (rollout cadence, data paths, perf args) from base_args.
    Sets policy_name = cfg.name so downstream code (update_weights routing,
    Sample.policy_name tagging) can read it from args.

    Builds a fresh namespace on every call. For Megatron-hosting policies, this
    also runs HF + Megatron validation; the Megatron validator intentionally
    mutates the namespace with its normalized defaults.
    """
    from argparse import Namespace

    ns = Namespace(**vars(base_args))
    for f in dataclasses.fields(cfg):
        setattr(ns, f.name, getattr(cfg, f.name))
    # Apply per-policy Megatron flag overrides (extra_megatron_args) as the
    # last step before slime-specific reconciliation below. Unknown YAML keys
    # in the `megatron:` block land here; setattr-ing them onto ns overrides
    # whatever the global CLI MODEL_ARGS pushed onto base_args. This is how
    # an N-policy run can host policies with different architectures.
    if cfg.extra_megatron_args:
        for k, v in cfg.extra_megatron_args.items():
            setattr(ns, k, v)
    ns.actor_num_nodes = cfg.megatron_num_nodes
    ns.actor_num_gpus_per_node = cfg.num_gpus_per_node
    ns.num_gpus_per_node = cfg.num_gpus_per_node
    ns.world_size = cfg.megatron_num_nodes * cfg.num_gpus_per_node
    ns.policy_name = cfg.name
    # Megatron's tokenizer fallback runs at parse_args time when global
    # hf_checkpoint may still be None. Re-derive per policy now that
    # cfg.hf_checkpoint is known.
    if getattr(ns, "tokenizer_model", None) is None:
        ns.tokenizer_model = cfg.hf_checkpoint
        if not getattr(ns, "tokenizer_type", None):
            ns.tokenizer_type = "HuggingFaceTokenizer"

    # Mirror slime_validate_args' per-policy-affecting defaults. Upstream
    # runs these once on global args at parse_args time; per-policy actors
    # only inherit values from their own cfg, so we re-apply the same logic
    # here. (Bugs of the form "policy A trains fine because eps_clip_high=0.28
    # was on the actor's args, but policy B crashes because its YAML omitted
    # eps_clip_high and ns.eps_clip_high stayed None.")
    if getattr(ns, "eps_clip_high", None) is None:
        ns.eps_clip_high = ns.eps_clip
    # n_samples_per_prompt == 1 means the GRPO group has size 1; the std is
    # zero and the group-norm divide produces NaN. Upstream forces the flag
    # off in that case.
    if getattr(ns, "n_samples_per_prompt", 1) == 1:
        ns.grpo_std_normalization = False
    # When dynamic batching is on, the log-probs forward pass uses the same
    # token budget as the train forward unless the user overrides it.
    if getattr(ns, "use_dynamic_batch_size", False):
        if getattr(ns, "log_probs_max_tokens_per_gpu", None) is None:
            ns.log_probs_max_tokens_per_gpu = ns.max_tokens_per_gpu

    # Re-run slime_validate_args' load resolution per policy. Upstream slime
    # runs this once at parse_args time against the GLOBAL args namespace, but
    # the globals don't see per-policy values (megatron_to_hf_mode / ref_load /
    # load all live in each policy's YAML entry). So we apply the exact same
    # logic — character-for-character — to the per-policy ns. No multi-policy-
    # specific shortcuts: behavior matches upstream when run with a single
    # policy, and each per-policy actor lands at the same args state it would
    # have reached through legacy CLI flags.
    import os

    if ns.megatron_to_hf_mode == "bridge":
        if (
            ns.load is not None
            and os.path.exists(ns.load)
            and os.path.exists(os.path.join(ns.load, "latest_checkpointed_iteration.txt"))
        ):
            # Megatron torch_dist ckpt at ns.load → resume from it; mbridge
            # only used in model_provider for the architecture spec.
            pass
        else:
            if ns.load is None:
                ns.load = ns.ref_load or cfg.hf_checkpoint
            # HF/bridge load → start at rollout 0
            ns.start_rollout_id = 0
    else:
        if (
            ns.load is None
            or not os.path.exists(ns.load)
            or not os.path.exists(os.path.join(ns.load, "latest_checkpointed_iteration.txt"))
        ):
            ns.no_load_optim = True
            ns.no_load_rng = True
            ns.finetune = True
            ns.load = ns.ref_load
            if getattr(ns, "ref_ckpt_step", None) is not None:
                ns.ckpt_step = ns.ref_ckpt_step
            ns.start_rollout_id = 0

    # Re-derive Megatron defaults whose inputs the policy may have
    # overridden (e.g. padded_vocab_size when vocab_size changed). Runs
    # last so it sees the final extras-projected values.
    _apply_megatron_defaults(ns, cfg)

    # For Megatron-hosting policies, run HF + structural validation on
    # this namespace. Megatron validation mutates `ns`
    # (variable_seq_lengths=True, moe_token_dispatcher_type rewrites,
    # ...), so each caller rebuilds a fresh namespace and reapplies the
    # same normalization. Replaces the prior policy_args_by_name
    # pass-through scheme.
    #
    # Gated on Megatron being importable so config-layer unit tests
    # (no Megatron backend installed) still work — config_to_namespace
    # becomes a no-op on the validation step in that environment.
    if cfg.megatron_num_nodes > 0:
        try:
            from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args
        except ModuleNotFoundError:
            megatron_validate_args = None
        if megatron_validate_args is not None:
            _validate_hf_per_policy(ns)
            megatron_validate_args(ns)

    return ns


def derive_policy_slices(configs: list[PolicyConfig], total_idxs: list[int], colocate: bool) -> dict[str, list[int]]:
    """Carve global placement-group indices into per-policy actor slices + a rollout slice.

    Pure function — no Ray. Mirrors the carving logic of create_placement_groups_multi
    (Step 7 in plan.md) so it can be unit-tested without a real placement group.

    Returns dict[name, list[int]] where:
      result[<policy_name>] = idxs assigned to that policy's Megatron actor
      result["rollout"]     = idxs for the rollout (shared with actors when colocate)
    """
    actor_gpus, rollout_gpus, total = derive_cluster_sizing(configs, colocate=colocate)
    if len(total_idxs) != total:
        raise ValueError(f"total_idxs has {len(total_idxs)} elements, derive_cluster_sizing wants {total}")

    cursor = 0
    result: dict[str, list[int]] = {}
    for c in configs:
        ss = c.megatron_num_nodes * c.num_gpus_per_node
        result[c.name] = list(total_idxs[cursor : cursor + ss])
        cursor += ss

    if colocate:
        result["rollout"] = list(total_idxs)
    else:
        result["rollout"] = list(total_idxs[cursor:])
    return result
