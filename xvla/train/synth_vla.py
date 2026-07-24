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
    obj_pairs = torch.zeros(bs, k_objects, 2, device=device, dtype=torch.long)  # (color,shape)/obj
    tgt_idx = torch.zeros(bs, device=device, dtype=torch.long)
    instr = torch.full((bs, instr_len), PAD, device=device, dtype=torch.long)
    for b in range(bs):
        pj = pairs[torch.randperm(16, device=device)[:k_objects]]
        cells = torch.randperm(GRID * GRID, device=device)[:k_objects]
        obj_pairs[b] = pj
        for o in range(k_objects):
            col, sh = int(pj[o, 0]), int(pj[o, 1])
            cy, cx = int(cells[o] // GRID), int(cells[o] % GRID)
            patch = colors[col][:, None, None] * masks[sh][None]
            img[b, :, cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL] = patch
            pos = torch.tensor([(cx + 0.5) / GRID, (cy + 0.5) / GRID], device=device)
            all_pos[b, o] = pos
        tgt = int(torch.randint(k_objects, (1,), device=device))
        tgt_idx[b] = tgt
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
                actions=actions, blind_mse=blind_mse,
                all_pos=all_pos, obj_pairs=obj_pairs, tgt_idx=tgt_idx)


def make_ambiguous_batch(bs, device, sep=0.5, instr_len=16, seed_offset=0):
    """Conditionally-BIMODAL reach: two IDENTICAL valid targets, one ambiguous instruction.

    Each scene places TWO objects with the SAME (colour, shape) — the pair named by the
    instruction "reach the {colour} {shape}" — at symmetric positions p_left, p_right about
    the image centre, separated (in normalized [0,1] target-coords) by `sep`. A third
    distractor object of a DIFFERENT pair is also placed. The instruction cannot pick which
    of the two identical objects is meant, so the target chunk is drawn 50/50 toward one of
    them: the SAME (image, instruction, state) admits TWO valid action chunks.

    This is the minimal task on which a mean/MSE head PROVABLY fails: the L2-optimal
    deterministic prediction is the mean = midpoint (p_left+p_right)/2 = empty space, which
    for sep>2*eps_hit reaches neither object. Returns p_left/p_right so the eval can score
    'commits to a valid mode' vs 'averages into the invalid middle'.
    """
    masks = _shape_masks(device)
    colors = COLORS.to(device)
    img = torch.zeros(bs, 3, 32, 32, device=device)
    instr = torch.full((bs, instr_len), PAD, device=device, dtype=torch.long)
    p_left = torch.zeros(bs, 2, device=device)
    p_right = torch.zeros(bs, 2, device=device)
    tgt_pos = torch.zeros(bs, 2, device=device)
    grip = torch.rand(bs, 2, device=device)
    H = 4
    half = sep / 2.0

    def _draw(b, col, sh, pos_xy):
        # pos_xy in [0,1]^2 → nearest grid cell for rendering
        cx = int(min(GRID - 1, max(0, round(float(pos_xy[0]) * GRID - 0.5))))
        cy = int(min(GRID - 1, max(0, round(float(pos_xy[1]) * GRID - 0.5))))
        patch = colors[col][:, None, None] * masks[sh][None]
        img[b, :, cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL] = patch

    for b in range(bs):
        col = int(torch.randint(N_COLORS, (1,), device=device))
        sh = int(torch.randint(N_SHAPES, (1,), device=device))
        # symmetric pair about centre along a random axis
        ang = float(torch.rand(1, device=device)) * 3.14159
        off = torch.tensor([half * torch.cos(torch.tensor(ang)),
                            half * torch.sin(torch.tensor(ang))], device=device)
        centre = torch.tensor([0.5, 0.5], device=device)
        pL = (centre - off).clamp(0.05, 0.95); pR = (centre + off).clamp(0.05, 0.95)
        p_left[b] = pL; p_right[b] = pR
        _draw(b, col, sh, pL); _draw(b, col, sh, pR)
        # distractor: a different (colour,shape) pair, placed off-centre
        dcol, dsh = (col + 1) % N_COLORS, (sh + 1) % N_SHAPES
        _draw(b, dcol, dsh, torch.tensor([0.5, 0.05], device=device))
        instr[b, :4] = torch.tensor([BOS, REACH, COLOR_TOK0 + col, SHAPE_TOK0 + sh], device=device)
        # 50/50 choose which identical target this demo reaches
        tgt_pos[b] = pL if float(torch.rand(1, device=device)) < 0.5 else pR

    disp = tgt_pos - grip
    step = disp / H
    actions = torch.zeros(bs, H, 7, device=device)
    actions[:, :, 0] = step[:, 0:1]
    actions[:, :, 1] = step[:, 1:2]
    actions[:, :, 6] = 1.0
    state = torch.zeros(bs, 8, device=device)
    state[:, :2] = grip
    embodiment = torch.zeros(bs, device=device, dtype=torch.long)
    return dict(img=img, instr=instr, state=state, embodiment=embodiment,
                actions=actions, p_left=p_left, p_right=p_right, tgt_pos=tgt_pos)
