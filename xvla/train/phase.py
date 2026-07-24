"""Cheap supervised discrete latent for unimodalizing a robot-action distribution.

Reframe (see DEVLOG "TENSOR-PURE multimodal action" thread): the LIBERO gripper
target is bimodal {-1,+1}; a linear+MSE head can only emit the conditional MEAN,
which averages the two modes to ~0 → the robot never commits to a grasp → 0%
closed-loop. The multimodality is not intrinsic to obs — it is an unobserved PHASE
(reach vs grasp vs lift). Condition on that phase z and p(a|obs,z) becomes
(near-)unimodal, so a linear+MSE (tensor-pure/foldable) head suffices again.

The demo GRIPPER SIGNAL itself labels the phase — a free supervised latent:
  - gripper action < 0  → OPEN  (approaching / reaching / after release)
  - gripper action > 0  → CLOSED (grasping / transporting / lifting)
and the OPEN→CLOSED / CLOSED→OPEN transitions bracket the grasp. This module turns
a chunk of gripper actions into per-timestep phase labels with no extra annotation.

All functions are numpy so they run in the data pipeline before tensors are built.
"""

from __future__ import annotations

import numpy as np


def gripper_sign_phase(gripper: np.ndarray, close_positive: bool = True) -> np.ndarray:
    """2-phase latent = sign of the (bimodal) gripper command.

    gripper: (...,) array of the gripper action dim, values ≈ {-1,+1}.
    returns int labels in {0:open, 1:closed}. This is the *minimal* latent: it
    makes the gripper dim conditionally DETERMINISTIC by construction (phase 0 →
    -1, phase 1 → +1), so gripper MSE|z ≈ 0 while marginal gripper MSE ≈ var(mode)."""
    g = np.asarray(gripper, dtype=np.float32)
    closed = g > 0 if close_positive else g < 0
    return closed.astype(np.int64)


def reach_grasp_lift_phase(gripper_seq: np.ndarray, close_positive: bool = True) -> np.ndarray:
    """3-phase latent {0:reach, 1:grasp/close, 2:lift/transport} for one EPISODE.

    Derived purely from the gripper command timeline:
      - before the first close  → REACH
      - the (short) window straddling the first sustained OPEN→CLOSE transition → GRASP
      - everything after the arm is committed-closed → LIFT/TRANSPORT
    This 3-way split further unimodalizes the ARM dims: during REACH the eef moves
    toward the object, during LIFT it moves up/away — at a given (img,state) those
    two arm directions are the bimodality that a phase-blind MSE head averages out.

    gripper_seq: (Teps,) gripper action over one episode (time-ordered).
    returns (Teps,) int labels."""
    g = np.asarray(gripper_seq, dtype=np.float32)
    closed = (g > 0) if close_positive else (g < 0)
    lab = np.zeros(len(g), dtype=np.int64)              # default REACH
    if closed.any():
        first_close = int(np.argmax(closed))            # first True index
        lab[first_close:] = 2                            # committed-closed → LIFT
        # a small grasp window around the transition
        w = 2
        lo, hi = max(0, first_close - 0), min(len(g), first_close + w)
        lab[lo:hi] = 1                                   # GRASP
    return lab


def label_chunks_sign(chunk_actions: np.ndarray, gripper_dim: int = -1,
                      close_positive: bool = True) -> np.ndarray:
    """Per-chunk 2-phase label from the FIRST action's gripper sign in each chunk.

    chunk_actions: (N, H, d_a) action chunks (the exact objects the head regresses).
    Uses the first step's gripper command as the phase in effect when the chunk is
    emitted. Returns (N,) int labels in {0,1}. This is the label to teacher-force
    at the phase-conditioning token during training."""
    g0 = chunk_actions[:, 0, gripper_dim]
    return gripper_sign_phase(g0, close_positive=close_positive)


def conditional_vs_marginal_spread(chunk_actions: np.ndarray, labels: np.ndarray,
                                   gripper_dim: int = -1) -> dict:
    """Quantify how much conditioning unimodalizes — a DATA-ONLY diagnostic (no model).

    Compares, for every action dim, the marginal std (what a phase-blind MSE head is
    forced to average over) against the label-conditional std (what a phase-conditioned
    head must fit). A large drop, especially on the gripper dim, is direct evidence
    that the residual conditional distribution is (near-)unimodal. Returns per-dim
    marginal std, mean within-phase std, and their ratio (unimodalization factor)."""
    A = np.asarray(chunk_actions)                       # (N, H, d_a)
    flat = A.reshape(-1, A.shape[-1])                   # (N*H, d_a)
    lab = np.repeat(np.asarray(labels), A.shape[1])     # (N*H,)
    marg = flat.std(0)
    within = np.zeros_like(marg)
    for z in np.unique(lab):
        m = lab == z
        within += m.mean() * flat[m].std(0)             # E_z[std(a|z)]
    ratio = marg / (within + 1e-9)
    return {"marginal_std": marg.tolist(),
            "within_phase_std": within.tolist(),
            "unimodalization_ratio": ratio.tolist(),
            "gripper_marginal_std": float(marg[gripper_dim]),
            "gripper_within_phase_std": float(within[gripper_dim]),
            "gripper_ratio": float(ratio[gripper_dim])}
