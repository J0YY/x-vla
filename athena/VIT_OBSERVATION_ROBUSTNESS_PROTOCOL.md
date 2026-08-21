# Frozen ViT paired observation-robustness protocol

This protocol was fixed before observing any robustness rollout. It is one bounded operational
robustness experiment, not a corruption sweep from which favorable conditions may be selected.

## Frozen artifacts and trials

The program uses the three SHA-256-bound ViT checkpoints and corrected Object cache named by the
immutable capability artifact. It validates completed cache provenance job 830988, the pinned
LeRobot revision and metadata, and the corrected dataset-to-official task map. Each checkpoint is
evaluated on canonical episodes zero through nine for all ten official tasks. Every episode is run
once clean and once under each of three fixed shifts, giving 1,200 closed-loop trials.

All conditions reset the same simulator state, use the same official instruction, run for at most
280 actions, settle for ten steps, and execute eight actions per receding-horizon prediction. Every
evaluation uses an NVIDIA RTX A6000 and highest matrix precision.

## Frozen camera-stream conditions

- `clean`: the deployed 180-degree-rotated, 64-by-64 RGB input.
- `brightness_0p85`: multiply RGB intensity by 0.85 after resize and clip to the valid range.
- `gaussian_4_over_255`: add deterministic independent RGB Gaussian noise with standard deviation
  4/255 after resize and clip to the valid range. The noise seed is derived from checkpoint, task,
  episode, decision, and condition identities.
- `translate_down_right_1px`: translate the resized image one pixel down and right with edge
  replication.

These are mild, synthetic camera calibration and noise shifts. They were chosen as a single fixed
set before outcomes. No condition may be removed, weakened, or promoted based on its result.

## Frozen primary gate

For each corruption, resample paired episodes within every one of the 30 fixed checkpoint-task
cells for 20,000 deterministic bootstrap draws. The experiment passes only if both conditions hold
for all three corruptions:

1. the 95 percent interval lower endpoint for shifted-to-clean success retention is at least 0.75
2. every checkpoint's point retention is at least 0.70

The bootstrap describes the finite canonical episode set. It does not estimate a population of
tasks or checkpoints.

## Claim boundary

A pass supports bounded closed-loop retention under these three prespecified mild camera-stream
shifts on familiar Object tasks, prompts, objects, scenes, and checkpoints. It is not an
adversarial-robustness benchmark, an architecture comparison, robust language grounding, or
evidence of unseen-task, cross-suite, physical-robot, or broad environmental generalization.
