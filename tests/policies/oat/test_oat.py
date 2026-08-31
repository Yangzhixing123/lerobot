import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.oat.action_tokenizer import FSQ, RFSQ, OATActionTokenizer
from lerobot.policies.oat_fsq.configuration_oat_fsq import OATFSQConfig
from lerobot.policies.oat_fsq.modeling_oat_fsq import OATFSQPolicy
from lerobot.policies.oat_rfsq_pair.configuration_oat_rfsq_pair import OATRFSQPairConfig
from lerobot.policies.oat_rfsq_pair.modeling_oat_rfsq_pair import OATRFSQPairPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(config_class):
    return config_class(
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        horizon=4,
        n_action_steps=2,
        latent_horizon=2,
        tokenizer_dim=32,
        tokenizer_encoder_layers=1,
        tokenizer_decoder_layers=1,
        tokenizer_heads=4,
        policy_dim=32,
        policy_layers=1,
        policy_heads=4,
        temperature=0,
    )


@pytest.mark.parametrize(
    ("policy_type", "config_class", "policy_class"),
    [
        ("oat_fsq", OATFSQConfig, OATFSQPolicy),
        ("oat_rfsq_pair", OATRFSQPairConfig, OATRFSQPairPolicy),
    ],
)
def test_factory_and_temporal_indices(policy_type, config_class, policy_class):
    assert get_policy_class(policy_type) is policy_class
    assert isinstance(make_policy_config(policy_type, device="cpu"), config_class)
    config = _config(config_class)
    assert config.observation_delta_indices == [-1, 0]
    assert config.action_delta_indices == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("config_class", "policy_class", "token_count"),
    [(OATFSQConfig, OATFSQPolicy, 2), (OATRFSQPairConfig, OATRFSQPairPolicy, 4)],
)
def test_forward_and_online_action_shape(config_class, policy_class, token_count):
    policy = policy_class(_config(config_class))
    batch = {
        OBS_STATE: torch.randn(2, 2, 5),
        ACTION: torch.randn(2, 4, 3),
    }
    tokens = policy._policy_tokens(batch[ACTION])
    assert tokens.shape == (2, token_count)
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert metrics["cross_entropy"] > 0
    assert not any(parameter.requires_grad for parameter in policy.action_tokenizer.parameters())

    action = policy.select_action({OBS_STATE: torch.randn(2, 5)})
    assert action.shape == (2, 3)
    assert len(policy.action_queue) == 1


def test_paired_rfsq_interleaves_complete_latent_positions():
    policy = OATRFSQPairPolicy(_config(OATRFSQPairConfig))
    stage_major = torch.tensor([[10, 11, 20, 21]])
    assert torch.equal(policy._stage_major_to_interleaved(stage_major), torch.tensor([[10, 20, 11, 21]]))


@pytest.mark.parametrize(
    ("config_class", "policy_class"),
    [(OATFSQConfig, OATFSQPolicy), (OATRFSQPairConfig, OATRFSQPairPolicy)],
)
def test_policy_checkpoint_is_self_contained(tmp_path, config_class, policy_class):
    policy = policy_class(_config(config_class))
    policy.save_pretrained(tmp_path)
    loaded = policy_class.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.config.action_tokenizer_path is None
    parameter_pairs = zip(policy.parameters(), loaded.parameters(), strict=True)
    assert all(torch.equal(left, right) for left, right in parameter_pairs)


def test_fsq_codebook_round_trip():
    quantizer = FSQ((4, 3, 2))
    indices = torch.arange(quantizer.codebook_size)
    codes = quantizer.indices_to_embedding(indices)
    assert torch.equal(quantizer.codes_to_indices(codes), indices)


def test_rfsq_quantized_latents_round_trip_from_stage_major_tokens():
    quantizer = RFSQ((4, 3, 2), latent_horizon=2)
    quantized, tokens = quantizer(torch.randn(3, 2, 3))
    restored = quantizer.indices_to_embedding(tokens)
    assert tokens.shape == (3, 4)
    assert torch.allclose(restored, quantized)


def test_prefix_decode_masks_tail_latents():
    config = _config(OATFSQConfig)
    tokenizer = OATActionTokenizer(config, action_dim=3, residual=False).eval()
    latents = torch.randn(2, config.latent_horizon, len(config.fsq_levels))
    changed_tail = latents.clone()
    changed_tail[:, 1:] += 100
    assert torch.allclose(
        tokenizer.decode(latents, keep_k=1),
        tokenizer.decode(changed_tail, keep_k=1),
    )
    with pytest.raises(ValueError, match="keep_k"):
        tokenizer.decode(latents, keep_k=0)


@pytest.mark.parametrize("config_class", [OATFSQConfig, OATRFSQPairConfig])
def test_default_processors_are_discoverable(config_class):
    preprocessor, postprocessor = make_pre_post_processors(_config(config_class), dataset_stats=None)
    assert preprocessor is not None
    assert postprocessor is not None


def test_tokenizer_preserves_causal_register_and_action_order():
    config = _config(OATFSQConfig)
    tokenizer = OATActionTokenizer(config, action_dim=3, residual=False).eval()
    actions = torch.randn(2, config.horizon, 3)
    with torch.inference_mode():
        baseline_latents = tokenizer.encode(actions)
        tokenizer.register_tokens[:, 1:].add_(100)
        changed_latents = tokenizer.encode(actions)
    assert torch.allclose(baseline_latents[:, 0], changed_latents[:, 0], atol=1e-5)

    latents = torch.randn(2, config.latent_horizon, len(config.fsq_levels))
    with torch.inference_mode():
        baseline_actions = tokenizer.decode(latents)
        tokenizer.action_queries[:, -1].add_(100)
        changed_actions = tokenizer.decode(latents)
    assert torch.allclose(baseline_actions[:, 0], changed_actions[:, 0], atol=1e-5)
