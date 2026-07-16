"""Procedural synthetic VLA task (Milestone 4a) — language is load-bearing.

Renders 32×32 scenes with K distinct (color, shape) objects on a grid; the
instruction "reach the {color} {shape}" selects one; the target action chunk moves
the gripper toward that object. Because a different object is named each episode,
the policy MUST read the instruction — quantified by a language-grounding gap
(MSE with shuffled instructions minus MSE with correct ones) and a computable
language-blind lower bound (always reach the object centroid).

All generated on-device in tensor ops — no downloads.
"""

from __future__ import annotations

import torch

COLORS = torch.tensor([[1, 0, 0], [0, 1, 0], [0.2, 0.4, 1.0], [1, 1, 0]])  # R G B Y
N_COLORS, N_SHAPES = 4, 4
GRID = 4          # 4x4 cells on a 32px image → 8px cells
CELL = 8
# Vocabulary: PAD BOS reach the | c0..c3 | s0..s3
PAD, BOS, REACH, THE = 0, 1, 2, 3
COLOR_TOK0, SHAPE_TOK0 = 4, 4 + N_COLORS
VOCAB = SHAPE_TOK0 + N_SHAPES


def _shape_masks(device):
    m = torch.zeros(N_SHAPES, CELL, CELL, device=device)
    yy, xx = torch.meshgrid(torch.arange(CELL, device=device),
                            torch.arange(CELL, device=device), indexing="ij")
    c = (CELL - 1) / 2
    m[0] = 1.0                                              # square
    m[1] = (((yy - c) ** 2 + (xx - c) ** 2) <= (c * 0.9) ** 2).float()   # circle
    m[2] = (yy >= xx).float()                               # triangle
    m[3] = (((yy - c).abs() <= 1) | ((xx - c).abs() <= 1)).float()       # cross
    return m


def make_batch(bs, device, k_objects=3, instr_len=16, shuffle_instr=False, seed_offset=0):
    masks = _shape_masks(device)
    colors = COLORS.to(device)
    img = torch.zeros(bs, 3, 32, 32, device=device)
    # choose k distinct (color, shape) pairs and k distinct cells per sample
    pairs = torch.stack(torch.meshgrid(torch.arange(N_COLORS, device=device),
                                       torch.arange(N_SHAPES, device=device),
                                       indexing="ij"), -1).reshape(-1, 2)  # 16 pairs
    tgt_pos = torch.zeros(bs, 2, device=device)
    all_pos = torch.zeros(bs, k_objects, 2, device=device)
    instr = torch.full((bs, instr_len), PAD, device=device, dtype=torch.long)
    for b in range(bs):
        pj = pairs[torch.randperm(16, device=device)[:k_objects]]
        cells = torch.randperm(GRID * GRID, device=device)[:k_objects]
        for o in range(k_objects):
            col, sh = int(pj[o, 0]), int(pj[o, 1])
            cy, cx = int(cells[o] // GRID), int(cells[o] % GRID)
            patch = colors[col][:, None, None] * masks[sh][None]
            img[b, :, cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL] = patch
            pos = torch.tensor([(cx + 0.5) / GRID, (cy + 0.5) / GRID], device=device)
            all_pos[b, o] = pos
        tgt = int(torch.randint(k_objects, (1,), device=device))
        tgt_pos[b] = all_pos[b, tgt]
        col, sh = int(pj[tgt, 0]), int(pj[tgt, 1])
        instr[b, :4] = torch.tensor([BOS, REACH, COLOR_TOK0 + col, SHAPE_TOK0 + sh], device=device)

    if shuffle_instr:
        instr = instr[torch.randperm(bs, device=device)]

    grip = torch.rand(bs, 2, device=device)
    H = 4
    disp = tgt_pos - grip
    step = disp / H
    actions = torch.zeros(bs, H, 7, device=device)
    actions[:, :, 0] = step[:, 0:1]
    actions[:, :, 1] = step[:, 1:2]
    actions[:, :, 6] = 1.0                                  # gripper "close" flag
    state = torch.zeros(bs, 8, device=device)
    state[:, :2] = grip
    embodiment = torch.zeros(bs, device=device, dtype=torch.long)

    # Language-blind lower bound: reach the object centroid (ignores instruction).
    centroid = all_pos.mean(1)
    blind_step = (centroid - grip) / H
    blind_actions = torch.zeros_like(actions)
    blind_actions[:, :, 0] = blind_step[:, 0:1]
    blind_actions[:, :, 1] = blind_step[:, 1:2]
    blind_actions[:, :, 6] = 1.0
    blind_mse = (blind_actions - actions).pow(2).mean().item()

    return dict(img=img, instr=instr, state=state, embodiment=embodiment,
                actions=actions, blind_mse=blind_mse)
