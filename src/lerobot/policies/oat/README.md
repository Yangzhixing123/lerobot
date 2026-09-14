# OAT policies in LeRobot

This port exposes two independent LeRobot policy types:

- `oat_fsq`: the original OAT route, with one FSQ token per latent position.
- `oat_rfsq_pair`: two-stage residual FSQ. The tokenizer stores stage-major tokens, while the policy
  trains and generates adjacent complete pairs: `q1_1, q2_1, q1_2, q2_2, ...`.

Both consume standard LeRobot policy features and dataset windows. The defaults use observations at
`[-1, 0]`, a 32-step action chunk, and execute 16 actions before replanning. Image tensors are the
LeRobot-native `C,H,W` format; action dimension and camera keys are inferred from dataset metadata.

## Training

Train a tokenizer first (the two policy routes require different tokenizer checkpoints):

```bash
lerobot-train-oat-tokenizer \
  --repo_id=my-org/my-dataset \
  --policy_type=oat_fsq \
  --output_dir=outputs/oat_fsq_tokenizer

lerobot-train-oat-tokenizer \
  --repo_id=my-org/my-dataset \
  --policy_type=oat_rfsq_pair \
  --output_dir=outputs/oat_rfsq_pair_tokenizer
```

Then run the regular LeRobot trainer:

```bash
lerobot-train \
  --dataset.repo_id=my-org/my-dataset \
  --policy.type=oat_fsq \
  --policy.action_tokenizer_path=outputs/oat_fsq_tokenizer \
  --output_dir=outputs/oat_fsq_policy
```

Use `--policy.type=oat_rfsq_pair` and its matching tokenizer directory for the paired route. The final
policy checkpoint embeds the frozen tokenizer weights, so deployment no longer depends on the separate
tokenizer directory. `policy.use_k_tokens=K` means K latent positions for both routes; paired RFSQ
internally generates `2*K` discrete tokens.

Multi-task datasets can set `policy.num_tasks` and `policy.task_names`; the policy accepts LeRobot's
`task_index` during training and maps evaluation task strings through `task_names`.

The FSQ/RFSQ implementation retains the upstream EPFL-Apple non-commercial license notice. Review that
license before redistribution or commercial deployment.
