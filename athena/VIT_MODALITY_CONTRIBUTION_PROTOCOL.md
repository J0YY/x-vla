# Frozen ViT modality-contribution audit protocol

This protocol was fixed before any audit outcome was observed. It does not alter training,
evaluation, checkpoints, or paper text.

## Frozen identities

- ViT checkpoints: the exact seed-0, seed-1, and seed-2 artifacts in
  `artifacts/vit_capability/manifest.json`, including their frozen SHA-256 identities.
- Cache: `libero_frames_100000_64.pkl`, SHA-256
  `053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662`.
- Cache provenance: passed Object provenance job `830988`, including the pinned LeRobot revision,
  metadata SHA, canonical-content SHA, and dataset-to-official task map.
- Source: the runner records the SHA-256 identity of itself, its strict summarizer, both Slurm
  wrappers, the launcher, this protocol, and every imported model/operator source.

## Source-token partition

The deployed sequence is checked against a forward-pre-hook on the first joint block. The four
groups are an exhaustive partition of all 107 positions:

| group | positions | deployed contents |
|---|---:|---|
| vision | `[0,64)` | 64 ViT patch tokens |
| instruction | `[64,97)` | learned BOS plus 32 encoded instruction positions |
| robot state | `[97,99)` | projected 8D robot state plus embodiment token |
| action query | `[99,107)` | eight causal action-query tokens |

BOS is assigned to instruction and embodiment to robot state because they are the corresponding
deployed conditioning constants. The learned output-projection bias is not assigned arbitrarily to
a modality. It is audited as a separate source-independent constant.

## Inputs and cases

For each checkpoint, rank every cache row within its official task by
`SHA256(protocol_id || checkpoint_sha256 || canonical_cache_record_sha256)`. The full audit selects
13 rows from official tasks 0 through 7 and 12 rows from tasks 8 and 9, for 128 inputs per
checkpoint and 384 total. Smoke uses one row from each official task.

Every selected input covers all eight joint blocks and twelve attention heads. The full program
therefore contains 3,072 module-input cases, 36,864 head-input cases, and 147,456
head-source-input records.

## Exact decomposition and frozen gate

For each action-query row, a head update is a sum over visible source tokens. Masking that sum by
the four frozen source slices gives four additive tensors whose sum must reconstruct the full head
update. Each tensor is then carried through the appropriate slice of the learned linear output
projection. Their sum reconstructs the pre-bias module update, and adding the separately recorded
output bias reconstructs the full module update.
The block's deployed scalar attention gain is then applied to the grouped reconstruction and the
full module output. This final comparison verifies the gain-scaled residual-branch update that is
actually added to each action-query row.

The only favorable gate is fixed at maximum relative L2 error at most `1e-6`, with all references,
reconstructions, contribution metrics, and prompt-characterization outputs finite. The gate covers
head group sums, projected pre-bias group sums, projected group sums plus bias, and agreement with
the deployed float32 attention implementation before and after the block's residual gain. There is
no contribution-magnitude, sign, cosine, dominance, or prompt-response threshold.

## Descriptive metrics and denominators

- Coherent energy fraction:
  `||sum_token c_source||_F^2 / sum_source ||sum_token c_source||_F^2`.
- Token energy fraction:
  `sum_visible_token ||c_token||_2^2 / sum_source sum_visible_token ||c_token||_2^2`.
- Per-source-token normalized energy fraction: divide each source's token-energy sum by its number
  of visible source-token incidences, then normalize those four means to sum to one.
- Signed projection fraction: `<c_source, full_update> / ||full_update||_F^2`.
- Cosine with the total: `<c_source, full_update> / (||c_source||_F ||full_update||_F)`.

The raw result records every numerator and denominator. The strict summary reports checkpoint,
layer, head, and layer-head distributions without applying a favorable magnitude threshold.

## Prompt permutation

The full program also takes the lowest-ranked observation from each official task and evaluates
all ten familiar official Object prompts while holding image and state fixed. It reports physical
action-chunk differences and cosines relative to the matched prompt. This is only a descriptive
same-observation characterization. It is not a grounding result, a causal intervention on task
success, or a zero-shot language evaluation.

## Claim boundary

Passing supports an exact, input-conditioned, additive source-token decomposition of every joint
attention head, module output, and gain-scaled residual update on the frozen inputs. It does not
decompose the ViT vision blocks,
FFNs, residual stack, final normalization, action head, or complete policy. Contribution magnitude
does not establish causal necessity, human-nameable features, instruction grounding, or robotic
generalization.
