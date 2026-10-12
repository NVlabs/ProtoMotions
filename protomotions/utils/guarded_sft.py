"""Valid-only SFT forward/loss with a fixed DDP optimization schedule."""
import torch
from tensordict import TensorDict
from protomotions.agents.common.supervision import compute_supervision_loss
from protomotions.utils.valid_samples import valid_moments


def guarded_optimization_step(agent, batch_dict):
    mask = batch_dict["physics_valid"].bool()
    count = mask.sum().to(dtype=torch.float32)
    global_count = agent.fabric.all_reduce(count.clone(), reduce_op="sum")
    if global_count.item() == 0:
        # Every rank skips together. Do not apply AdamW decay or optimizer momentum.
        return {"physics_guard/skipped_empty_batch": count.new_tensor(1.),
                "physics_guard/valid_samples": count.detach()}
    data = {k: v[mask] for k,v in batch_dict.items() if k != "physics_valid"}
    if count.item() == 0:
        # A dummy keeps the same DDP graph/collectives, but contributes no moments or gradient.
        data = {k: torch.zeros_like(v[:1]) for k,v in batch_dict.items() if k != "physics_valid"}
    forward_valid = torch.full((len(data["action"]),), bool(count.item()), device=mask.device)
    with valid_moments(forward_valid):
        td = agent.training_model(TensorDict(data, batch_size=len(data["action"])))
    supervised, metrics = compute_supervision_loss(td, agent.config.loss)
    actions = td.get("privileged_action", td["action"])
    extra, extra_metrics = agent.calculate_extra_loss(td, actions)
    model_loss, model_metrics = agent.model.compute_model_loss(
        td, current_epoch=agent.current_epoch, zero_loss=supervised, log_prefix="model")
    local_loss = supervised + extra + model_loss
    metrics.update(extra_metrics); metrics.update(model_metrics)
    # DDP averages gradients across ranks. Weight by the true global valid count.
    scale = agent.fabric.world_size * count / global_count
    loss = local_loss * scale
    metrics = {key: value * scale for key, value in metrics.items()}
    metrics.update(agent._step_optimizer(loss=loss, model=agent.training_model,
                   optimizer=agent.supervised_optimizer, model_name="model"))
    metrics["losses/supervised_loss"] = loss.detach()
    metrics["physics_guard/valid_samples"] = count.detach()
    metrics["physics_guard/skipped_empty_batch"] = count.new_tensor(0.)
    metrics["train/ce_loss"] = (supervised * scale).detach()
    return metrics
