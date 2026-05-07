from __future__ import annotations

import logging

import ray

from slime.ray.placement_group import (
    create_placement_groups_multi,
    create_rollout_manager_multi,
    create_training_models_multi,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.policy_config import build_sglang_config_from_policies, derive_cluster_sizing, parse_policy_configs

logger = logging.getLogger(__name__)


# Multi-policy slime driver. Each policy in --config <path>.yaml gets its own
# Megatron actor paired 1:1 with an SGLang engine; frozen policies (e.g. OPD
# teachers) skip the engine and run forward-only, feeding external_data to
# trainable consumers. Replaces train.py for runs with N>=1 trainable actors.
def _set_multi_policy_global_defaults(args, policy_configs, actor_gpus: int, rollout_gpus: int) -> None:
    """Populate legacy global args that shared rollout code still reads.

    Per-policy actors get their own namespaces later (config_to_namespace).
    The values written here are cluster-level: --num-gpus-per-node MUST stay
    as physical GPUs-per-node, not the per-policy slice size from
    cfg.num_gpus_per_node (different concepts despite the name overlap).
    """
    if not policy_configs:
        raise ValueError("multi-policy config must contain at least one policy")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = policy_configs[0].hf_checkpoint
    # tokenizer_model fell back to None when hf_checkpoint was None at parse_args
    # time. Re-derive now that we know it.
    if getattr(args, "tokenizer_model", None) is None:
        args.tokenizer_model = args.hf_checkpoint
        if not getattr(args, "tokenizer_type", None):
            args.tokenizer_type = "HuggingFaceTokenizer"

    # All policies must agree on slice size in v1.
    slice_sizes = {cfg.num_gpus_per_node for cfg in policy_configs}
    if len(slice_sizes) != 1:
        raise ValueError("all policies must use the same num_gpus_per_node in v1; got " f"{sorted(slice_sizes)}")

    args.rollout_num_gpus = rollout_gpus
    args.megatron_total_gpus = actor_gpus

    # Cluster-level args.num_gpus_per_node = physical GPUs per node (from CLI).
    # Synthesize actor_num_nodes / actor_num_gpus_per_node so their product
    # equals actor_gpus while respecting the physical-node limit. Downstream
    # code (sglang_engine.get_base_gpu_id, weight_updater range checks) reads
    # these as cluster-level totals.
    phys_per_node = args.num_gpus_per_node
    if actor_gpus <= phys_per_node:
        args.actor_num_nodes = 1
        args.actor_num_gpus_per_node = actor_gpus
    elif actor_gpus % phys_per_node == 0:
        args.actor_num_nodes = actor_gpus // phys_per_node
        args.actor_num_gpus_per_node = phys_per_node
    else:
        raise ValueError(
            f"actor_gpus ({actor_gpus}) is not <= or a multiple of "
            f"--num-gpus-per-node ({phys_per_node}). Pass a --num-gpus-per-node "
            f"that divides actor_gpus, or adjust the per-policy num_gpus_per_node."
        )

    # Manager reads global args.n_samples_per_prompt / args.global_batch_size
    # for legacy single-buffer code paths. Multi-policy actually drives those
    # from the per-policy args, but mirroring the first policy here prevents
    # silent CLI/config drift.
    args.n_samples_per_prompt = policy_configs[0].n_samples_per_prompt
    args.global_batch_size = policy_configs[0].global_batch_size


def train(args):
    configure_logger()

    # parse the YAML config and validate v1 restrictions
    if not getattr(args, "config", None):
        raise ValueError("train_multi_policy.py requires --config <path>.yaml")
    policy_configs = parse_policy_configs(args.config)
    n_trainable = sum(1 for cfg in policy_configs if cfg.trainable)

    if args.check_weight_update_equal and n_trainable > 1:
        raise ValueError(
            "--check-weight-update-equal not supported with multiple policies "
            "(uses _get_updatable_server which assumes a single trainable server)"
        )
    if args.use_fault_tolerance and n_trainable > 1:
        raise ValueError("--use-fault-tolerance not supported with multiple policies in v1")

    # derive cluster sizing and surface it on args for downstream code
    actor_gpus, rollout_gpus, total_gpus = derive_cluster_sizing(policy_configs, colocate=args.colocate)
    _set_multi_policy_global_defaults(args, policy_configs, actor_gpus, rollout_gpus)
    logger.info(
        f"cluster sizing (colocate={args.colocate}): "
        f"actor_gpus={actor_gpus}, rollout_gpus={rollout_gpus}, total={total_gpus}"
    )

    # allocate per-policy actor slices + a rollout slice
    pgs = create_placement_groups_multi(args, policy_configs)
    init_tracking(args)

    # build the SglangConfig from per-policy sglang sub-blocks and start the manager
    sglang_config = build_sglang_config_from_policies(policy_configs)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager_multi(args, pgs["rollout"], sglang_config)

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # build N RayTrainGroups, register each with the manager, async_init, and
    # reconcile start_rollout_ids across policies. Returns dict[name, PolicyHandle].
    handles = create_training_models_multi(args, pgs, rollout_manager, policy_configs)

    # Always push trainable actor weights to rollout once weights are loaded.
    # Frozen producers have no paired engine; skip.
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    for h in handles.values():
        if h.config.trainable:
            h.train_group.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # eval-only path (no training rollouts)
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # train loop
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        # generate: returns dict[policy_name | "__shared__", list[batch_per_dp]]
        rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        # Train in two passes. Frozen producers (e.g. OPD Megatron teacher)
        # run forward-only first on the trainable policy's rollout data;
        # their returned dicts merge into a single external_data passed to
        # all trainable consumers. With no frozen producers this collapses
        # to a single-pass loop with external_data=None — bit-identical to
        # pre-multi-producer behavior.
        trainable_handles = [h for h in handles.values() if h.config.trainable]
        frozen_handles = [h for h in handles.values() if not h.config.trainable]

        # frozen producers first; seed from the first trainable's rollout data
        producer_outputs: dict[str, dict] = {}
        if frozen_handles:
            seed_name = trainable_handles[0].config.name
            seed_data = rollout_data.get(seed_name) or rollout_data.get("__shared__")
            if seed_data is None:
                logger.warning(
                    f"frozen producers got no rollout data; skipping {[h.config.name for h in frozen_handles]}"
                )
            else:
                for h in frozen_handles:
                    # async_train returns a list (one entry per worker in the
                    # train group). On a single PP=1 actor the list has one
                    # entry; on PP>1 only the last-PP-rank entry is populated
                    # and the rest are {}. Pick the first non-empty dict.
                    results = ray.get(
                        h.train_group.async_train(rollout_id, seed_data, external_data=None)
                    )
                    producer_outputs[h.config.name] = next((r for r in results if r), {})

        # merge producer outputs (last-write-wins on key collision); pass None when empty
        merged_external: dict = {}
        for out in producer_outputs.values():
            if out:
                merged_external.update(out)
        external_for_trainable = merged_external if merged_external else None

        # trainable consumers; serialized so policies sharing GPU slots don't conflict
        for h in trainable_handles:
            data = rollout_data.get(h.config.name) or rollout_data.get("__shared__")
            if data is None:
                logger.warning(f"policy {h.config.name} got no rollout data")
                continue
            ray.get(h.train_group.async_train(rollout_id, data, external_data=external_for_trainable))

        # save per-policy checkpoints (skipped when h.config.save is unset, as for frozen producers)
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            for h in handles.values():
                if h.config.save:
                    h.train_group.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # memory hygiene
        if not args.offload_train:
            for h in handles.values():
                h.train_group.clear_memory()

        # weight update — serialized; sglang engines on shared GPUs cannot accept
        # two simultaneous broadcasts. Frozen producers have no engine; skip.
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        for h in handles.values():
            if h.config.trainable:
                h.train_group.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # eval (routes through one trainable policy by convention)
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
