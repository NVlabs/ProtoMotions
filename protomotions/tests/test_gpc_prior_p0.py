"""P0 correctness and state-isolation tests, runnable without Isaac or a GPU."""
import math
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from protomotions.agents.common.latent import TARGET_LATENT_KEY
from protomotions.agents.evaluators.gpc_prior_metrics import (
    token_sufficient_statistics, summarize_token_statistics,
    token_histogram, evaluate_token_distribution,
)
from protomotions.agents.evaluators.gpc_prior_offline import evaluate_prior_offline
from protomotions.agents.evaluators.gpc_prior_evaluator import GPCPriorEvaluator
from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig


def test_masked_nll_is_global_token_weighted_unsmoothed():
    logits = torch.tensor([[[math.log(3), 0.], [float('nan'), float('nan')]],
                           [[0., math.log(3)], [0., 0.]]])
    targets = torch.tensor([[0, -100], [0, 1]])
    valid = targets >= 0
    result = summarize_token_statistics(token_sufficient_statistics(logits, targets, valid))
    assert result['val_nll'] == pytest.approx((-math.log(.75)-math.log(.25)+math.log(2))/3)
    assert result['token_top1_acc'] == pytest.approx(1/3)
    assert result['token_top5_acc'] == 1
    assert result['token_acc_pos_0'] == .5
    with pytest.raises(ValueError, match='No valid'):
        summarize_token_statistics(token_sufficient_statistics(logits, targets, torch.zeros_like(valid)))


def test_distributions_zero_bins_large_vocab_and_count_reduction():
    gt = torch.tensor([[0, 1], [1, 1]])
    counts = token_histogram(gt, 59049)
    same = evaluate_token_distribution(counts, counts)
    assert same['token_js_divergence'] == 0
    assert same['sample_token_count'] == 4
    assert same['effective_vocab'] == pytest.approx(1.5)
    assert same['token_utilization'] == pytest.approx(1.5/59049)
    a = token_histogram(torch.zeros(2, 2, dtype=torch.long), 6)
    b = token_histogram(torch.ones(2, 2, dtype=torch.long), 6)
    assert evaluate_token_distribution(a, b)['token_js_divergence'] == pytest.approx(math.log(2))
    combined = evaluate_token_distribution(a+b, a+b)
    assert combined['effective_vocab'] == pytest.approx(2.)
    assert evaluate_token_distribution(a,a)['effective_vocab'] == 1.
    mask = torch.tensor([[True, False], [False, True]])
    assert token_histogram(gt, 59049, mask).sum() == 2


class FakePrior(torch.nn.Module):
    logits_key = 'logits'
    generated_tokens_key = 'sampled'
    def forward(self, td):
        assert 'tokens' in td
        td['logits'] = torch.zeros(len(td), 2, 7)
        return td
    def generate(self, td, **kwargs):
        assert 'tokens' not in td and TARGET_LATENT_KEY not in td
        td['sampled'] = torch.randint(7, (len(td), 2))
        return td


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prior = FakePrior()
        self.latent_tokenization = SimpleNamespace(num_prior_tokens=2, prior_token_vocab_size=7,
                                                   fsq_scalars_per_prior_token=1)
    def _prior_tensordict(self, td, targets=None):
        data = {'state': td['state']}
        if targets is not None:
            data['tokens'] = targets
        return TensorDict(data, batch_size=td.batch_size)


def test_offline_reproducibility_rng_mode_and_free_generation():
    model = FakeModel()
    batches = [TensorDict({'state': torch.ones(12, 3), TARGET_LATENT_KEY: torch.zeros(12, 2, dtype=torch.long)}, [12])]
    rng = torch.get_rng_state().clone()
    first = evaluate_prior_offline(model, batches, device=torch.device('cpu'), seed=87)
    second = evaluate_prior_offline(model, batches, device=torch.device('cpu'), seed=87)
    assert first[0] == second[0]
    assert torch.equal(first[2], second[2])
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    assert first[0]['val_nll'] == pytest.approx(math.log(7))


def test_terminal_survival_allows_recovery_without_zeroing_actions_or_reset():
    class Model:
        calls = 0
        def generate(self, td):
            self.calls += 1
            assert TARGET_LATENT_KEY not in td
            return torch.ones(2, 3), {}
    model = Model()
    env = SimpleNamespace(dt=.1, contact_body_ids=torch.tensor([], dtype=torch.long))
    evaluator = GPCPriorEvaluator(SimpleNamespace(env=env, model=model, num_envs=2),
        SimpleNamespace(device=torch.device('cpu')), GPCPriorEvaluatorConfig(max_eval_steps=5, failure_persistence_steps=1))
    resets, actions = [], []
    evaluator._reset_cohort = lambda ids: resets.append(ids) or 'obs'
    evaluator._obs = lambda obs, n: TensorDict({'state': torch.ones(n, 1)}, [n])
    evaluator._step = lambda action, n: (actions.append(action.clone()) or 'obs', None, torch.ones(n, dtype=torch.bool), None, None)
    state = SimpleNamespace(rigid_body_contacts=torch.zeros(2, 1, dtype=torch.bool), rigid_body_vel=torch.zeros(2, 1, 3))
    evaluator._physical_state = lambda n: (state, torch.tensor([model.calls in (2,3), False]))
    metrics, details = evaluator.evaluate_prior_rollout(torch.tensor([0, 1]))
    assert metrics['survival_rate'] == 1.
    assert metrics['never_fell_rate'] == .5
    assert len(resets) == 1 and all(torch.all(a == 1) for a in actions)
    assert details['recovery_success_rate'] is None


@pytest.mark.parametrize('kwargs', [{'top_p': 0}, {'temperature': 0}, {'mode': 'invalid'}, {'eval_metrics_every': 0}])
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        GPCPriorEvaluatorConfig(**kwargs).validate()


def test_offline_invalid_padded_targets_never_enter_embedding():
    model = FakeModel()
    original_forward = model.prior.forward
    def checked(td):
        assert (td['tokens'] >= 0).all()
        return original_forward(td)
    model.prior.forward = checked
    batch = TensorDict({'state': torch.ones(2,3), TARGET_LATENT_KEY: torch.tensor([[0,-100],[1,2]]),
                        'valid_token_mask': torch.tensor([[True,False],[True,True]])}, [2])
    metrics, gt, sampled = evaluate_prior_offline(model, [batch], device=torch.device('cpu'))
    assert metrics['gt_token_count'] == metrics['sample_token_count'] == 3


def test_explicit_disable_and_reenable_cli():
    import argparse
    from protomotions.agents.evaluators.gpc_prior_cli import add_prior_eval_arguments, explicit_prior_eval_options
    parser = argparse.ArgumentParser()
    add_prior_eval_arguments(parser)
    disabled = explicit_prior_eval_options(parser.parse_args(['--prior-eval-disable']))
    assert disabled['enabled'] is False and disabled['eval_metrics_every'] is None
    enabled = explicit_prior_eval_options(parser.parse_args(['--gpc-prior-eval', '--prior-eval-every', '12']))
    assert enabled['enabled'] is True and enabled['eval_metrics_every'] == 12


def test_sampling_restored_even_on_failure():
    model = torch.nn.Linear(2,2)
    model.temperature, model.top_p = .7, .8
    restored = []
    env = SimpleNamespace(motion_lib='lib', motion_manager='manager',
                          robot_config=SimpleNamespace(reset_noise='noise'),
                          simulator=SimpleNamespace(_push_enabled=True),
                          save_state=lambda: 'state', restore_state=lambda state: restored.append(state))
    agent = SimpleNamespace(env=env, model=model, motion_lib='lib', eval=model.eval)
    evaluator = GPCPriorEvaluator(agent, SimpleNamespace(device=torch.device('cpu'), global_rank=0),
                                  GPCPriorEvaluatorConfig(temperature=1., top_p=.9))
    model.weight.grad = torch.ones_like(model.weight)
    weight, grad, rng = model.weight.clone(), model.weight.grad.clone(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with evaluator._evaluation_scope():
            assert model.temperature == 1. and model.top_p == .9
            torch.rand(7)
            raise RuntimeError('fail')
    assert model.temperature == .7 and model.top_p == .8 and model.training
    assert torch.equal(model.weight, weight) and torch.equal(model.weight.grad, grad)
    assert torch.equal(torch.get_rng_state(), rng) and restored == ['state']
