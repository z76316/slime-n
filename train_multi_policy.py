from __future__ import annotations

import argparse
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
from slime.utils.policy_config import (
    build_sglang_config_from_policies,
    config_to_namespace,
    derive_cluster_sizing,
    has_sglang_engine,
    parse_policy_configs,
    populate_rollout_arch_fields,
)

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


def _preparse_config() -> str:
    """Extract --config path from sys.argv without touching the real parser.

    Returns the path string or raises if absent — multi-policy requires it.
    """
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--config", type=str, default=None)
    ns, _ = p.parse_known_args()
    if not ns.config:
        raise ValueError("train_multi_policy.py requires --config <path>.yaml")
    return ns.config


def parse_multi_policy_args():
    """Parse CLI + YAML for multi-policy training.

    Returns (base_args, policy_configs). Per-policy validation (HF +
    Megatron structural) is done inside `config_to_namespace` itself, so
    every caller — the pre-flight pass below, the actor-construction
    call in `placement_group.py`, and the megatron-less placeholder in
    `train()` — gets a fully validated namespace. The pre-flight pass
    here exists only to derive rollout arch fields from engine-hosting
    policies (and to fail-fast on bad per-policy arch BEFORE Ray spawns
    actors).
    """
    config_path = _preparse_config()
    policy_configs = parse_policy_configs(config_path)

    # Defer global Megatron structural / HF validation — arch is
    # per-policy YAML, not global CLI. Per-policy validation happens
    # inside `config_to_namespace`.
    base_args = parse_args(skip_megatron_model_validation=True)

    # Set multi-policy globals on base_args BEFORE building per-policy
    # namespaces, since each namespace inherits fields from base_args.
    actor_gpus, rollout_gpus, _ = derive_cluster_sizing(policy_configs, colocate=base_args.colocate)
    _set_multi_policy_global_defaults(base_args, policy_configs, actor_gpus, rollout_gpus)

    # Pre-flight build: validates each per-policy namespace (inside
    # config_to_namespace) and surfaces num_layers etc. for the rollout
    # arch derivation. The namespaces themselves are discarded —
    # placement_group rebuilds via config_to_namespace at actor
    # construction time, which validates again. Validators are pure
    # functions of the namespace; double-run is cheap and removes the
    # need to plumb a `policy_args_by_name` dict through.
    all_policy_args = [config_to_namespace(cfg, base_args) for cfg in policy_configs]
    populate_rollout_arch_fields(base_args, policy_configs, all_policy_args)

    return base_args, policy_configs


def train(args, policy_configs):
    configure_logger()

    n_trainable = sum(1 for cfg in policy_configs if cfg.trainable)

    if args.check_weight_update_equal and n_trainable > 1:
        raise ValueError(
            "--check-weight-update-equal not supported with multiple policies "
            "(uses _get_updatable_server which assumes a single trainable server)"
        )
    if args.use_fault_tolerance and n_trainable > 1:
        raise ValueError("--use-fault-tolerance not supported with multiple policies in v1")

    # cluster sizing already populated by parse_multi_policy_args; log it.
    logger.info(
        f"cluster sizing (colocate={args.colocate}): "
        f"actor_gpus={args.megatron_total_gpus}, rollout_gpus={args.rollout_num_gpus}"
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
    # Pre-filter out megatron-less policies (m✗ s✓ frozen standalone engines
    # like OPD SGLang teacher / judge / RM) so create_training_models_multi
    # doesn't try to allocate a RayTrainGroup(num_nodes=0) for them. Those
    # policies' SGLang servers were already spawned via the sglang_config
    # path; we just need a PolicyHandle placeholder so per-handle iterations
    # in the train loop can find them.
    megatron_cfgs = [c for c in policy_configs if c.megatron_num_nodes > 0]
    handles = create_training_models_multi(args, pgs, rollout_manager, megatron_cfgs)
    from slime.utils.policy_config import PolicyHandle

    for cfg in policy_configs:
        if cfg.megatron_num_nodes == 0:
            handles[cfg.name] = PolicyHandle(config=cfg, args=config_to_namespace(cfg, args), train_group=None)

    # Always push trainable actor weights to rollout once weights are loaded.
    # Skip policies that don't host an engine (frozen Megatron teachers,
    # PPO critics) — they have no paired SGLang server to push to.
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    for h in handles.values():
        if h.config.trainable and has_sglang_engine(h.config):
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

        # Train in three passes, partitioned by shape (engine vs. no engine):
        #   1. frozen               — frozen Megatron producers (OPD teacher).
        #   2. trainable_standalone — trainable Megatron-only producers (PPO
        #                             critic today). Output feeds trainable_pair
        #                             via external_data.
        #   3. trainable_pair       — trainable actors with paired SGLang engines;
        #                             consume merged_external from the two
        #                             producer passes.
        # Pass order is load-bearing: pass 2's outputs must land in
        # producer_outputs before pass 3 reads them. Within each pass, all
        # producers run concurrently on the same seed_data. With no producers
        # in a bucket, that pass is a no-op — for every existing multi-policy
        # example (none has a trainable Megatron-only policy), the standalone
        # bucket is empty and the loop collapses to today's two-pass behavior,
        # bit-identical.
        #
        # producer_outputs[name] is the per-rank list (length = producer's
        # world_size), not a single collapsed dict — required so each
        # trainable rank gets its own producer rank's output (e.g. teacher
        # rank r's teacher_log_probs lands at student rank r). Collapsing to
        # "first non-empty" worked when DP=1 but silently broke alignment
        # with DP>1 (student rank N would receive teacher rank 0's shard).
        frozen_handles = [h for h in handles.values() if not h.config.trainable and h.train_group is not None]
        trainable_standalone = [h for h in handles.values() if h.config.trainable and not has_sglang_engine(h.config)]
        trainable_pair = [h for h in handles.values() if h.config.trainable and has_sglang_engine(h.config)]

        # Resolve producer seed data: prefer the first paired policy's rollout
        # slice (so producers see the same prompts the actor will train on),
        # fall back to __shared__ for shared-buffer mode.
        seed_name = trainable_pair[0].config.name if trainable_pair else None
        seed_data = rollout_data.get(seed_name) if seed_name else None
        if seed_data is None:
            seed_data = rollout_data.get("__shared__")

        producer_outputs: dict[str, list[dict]] = {}
        for producer_list, label in (
            (frozen_handles, "frozen"),
            (trainable_standalone, "trainable_standalone"),
        ):
            if not producer_list:
                continue
            if seed_data is None:
                logger.warning(
                    f"{label} producers got no rollout data; skipping " f"{[h.config.name for h in producer_list]}"
                )
                continue
            # Launch all producers in this pass concurrently; preserve per-rank
            # output so the trainable consumers can pull the right shard
            # (PP>1: only the last-PP-rank entry is populated, rest are {};
            # the consumer's actor.train ignores empty external_data).
            per_policy_refs: dict[str, list] = {
                h.config.name: h.train_group.async_train(rollout_id, seed_data, external_data=None)
                for h in producer_list
            }
            for name, refs in per_policy_refs.items():
                producer_outputs[name] = ray.get(refs)

        # Pass 3: trainable_pair — parallel: each policy lives on its own GPU(s)
        # under both colocate (M+S share one GPU) and no-colocate (separate
        # GPUs). Cross-policy concurrent train has disjoint NCCL groups,
        # disjoint optimizer state, and no shared mutable state — every actor
        # is its own Ray process.
        #
        # external_data is built per-consumer as a list[dict] (one dict per
        # consumer worker), where each rank's dict merges same-rank outputs
        # from every frozen producer. RayTrainGroup.async_train already
        # supports the list[dict] signature for per-rank routing. With no
        # frozen producers it stays None (single-pass loop, byte-identical to
        # pre-multi-producer behavior). After PP filtering, the producer's
        # effective "DP world" (the non-empty last-PP-stage ranks, in DP
        # order under Megatron's default tp-cp-ep-dp-pp mesh) must equal
        # consumer.world_size.
        in_flight: list = []
        for h in trainable_pair:
            data = rollout_data.get(h.config.name) or rollout_data.get("__shared__")
            if data is None:
                logger.warning(f"policy {h.config.name} got no rollout data")
                continue
            consumer_external = None
            if producer_outputs:
                num_ranks = len(h.train_group._actor_handlers)
                per_rank: list[dict] = [{} for _ in range(num_ranks)]
                for name, results in producer_outputs.items():
                    # PP-aware: producers with PP>1 emit logprobs only on
                    # last-PP-stage ranks; the rest return {}. Megatron's
                    # default --order tp-cp-ep-dp-pp puts last-PP-stage ranks
                    # contiguous at the tail in DP rank order, so filtering
                    # non-empty preserves the DP→consumer-rank mapping.
                    non_empty = [out for out in results if out]
                    if len(non_empty) != num_ranks:
                        raise RuntimeError(
                            f"frozen producer {name!r} has {len(non_empty)} non-empty rank outputs "
                            f"(of {len(results)} total ranks) but trainable consumer "
                            f"{h.config.name!r} has {num_ranks} ranks. Producer's last-PP-stage "
                            "DP must match consumer's DP."
                        )
                    for r, out in enumerate(non_empty):
                        per_rank[r].update(out)
                if any(per_rank):
                    consumer_external = per_rank
            # async_train returns list[ref] (one per worker); flatten with extend.
            in_flight.extend(h.train_group.async_train(rollout_id, data, external_data=consumer_external))
        if in_flight:
            ray.get(in_flight)

        # save per-policy checkpoints (skipped when h.config.save is unset, as for frozen producers)
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            for h in handles.values():
                if h.config.save:
                    h.train_group.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        # memory hygiene — parallel across policies (each actor's clear_memory
        # frees only its own GPU; no cross-actor coordination). Engine-only
        # policies have no Megatron actor (train_group is None) and nothing
        # to clear.
        if not args.offload_train:
            cleanup_refs: list = []
            for h in handles.values():
                if h.train_group is None:
                    continue
                cleanup_refs.extend(h.train_group.async_clear_memory())
            if cleanup_refs:
                ray.get(cleanup_refs)

        # weight update — kept serial: per-policy NCCL groups don't share state,
        # but slime/ray/rollout.py:410 allocates a *single* rollout_engine_lock
        # acquired by every policy's update_weights for the entire broadcast
        # (slime/.../update_weight_from_distributed.py:232-248). Concurrent calls
        # would spin-poll the same lock and end up serial with extra overhead.
        # Train phase dominates push 10-50× per rollout; not worth fixing here.
        # Frozen producers have no engine; skip.
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        for h in handles.values():
            if h.config.trainable and has_sglang_engine(h.config):
                h.train_group.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        # eval (routes through one trainable policy by convention)
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args, policy_configs = parse_multi_policy_args()
    train(args, policy_configs)
