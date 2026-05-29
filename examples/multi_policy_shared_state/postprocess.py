import torch


def _sample_scalar(advantage: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor | None:
    if advantage.numel() == 0 or loss_mask.numel() == 0:
        return None
    active = loss_mask.to(dtype=torch.bool)
    if active.sum() == 0:
        return None
    return advantage[active].float().mean()


def rollout_data_postprocess(args, rollout_id, rollout_data):
    if getattr(args, "policy_name", None) not in ("peer_a", "peer_b"):
        return
    if "round_number" not in rollout_data or "advantages" not in rollout_data:
        return

    group_size = args.n_samples_per_prompt
    if group_size != 12:
        return

    advantages = rollout_data["advantages"]
    loss_masks = rollout_data["loss_masks"]
    round_numbers = rollout_data["round_number"]
    if len(advantages) % group_size != 0:
        return

    for start in range(0, len(advantages), group_size):
        end = start + group_size
        for round_number in (1, 2, 3):
            indices = [i for i in range(start, end) if round_numbers[i] == round_number]
            scalars = []
            active_indices = []
            for i in indices:
                scalar = _sample_scalar(advantages[i], loss_masks[i])
                if scalar is None:
                    continue
                scalars.append(scalar)
                active_indices.append(i)
            if not scalars:
                continue

            values = torch.stack(scalars)
            values = values - values.mean()
            if getattr(args, "grpo_std_normalization", False) and values.numel() > 1:
                values = values / (values.std(unbiased=True) + 1e-6)

            for i, value in zip(active_indices, values, strict=False):
                advantages[i] = torch.ones_like(advantages[i], dtype=advantages[i].dtype) * value.to(
                    dtype=advantages[i].dtype
                )

    rollout_data["advantages"] = advantages
    if getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]:
        rollout_data["returns"] = [adv.clone() for adv in advantages]
