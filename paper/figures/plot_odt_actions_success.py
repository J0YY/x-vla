"""Render the audited 20-start action/rollout comparison as vector artwork.

Only reads existing results. No model or decomposition code is imported.
Run with the bundled Python runtime from any directory.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'output/odt_campaign_20260907/rollout'
OUTPUT = ROOT / 'output/figures'
PDF = ROOT / 'output/pdf/odt_actions_success.pdf'
SUMMARY = json.loads((EVIDENCE / 'summary.json').read_text())
ENTRY = json.loads((EVIDENCE / 'entry_descriptive_v1.json').read_text())
PAIRS_PATH = EVIDENCE / 'paired_episodes.npz'
assert hashlib.sha256(PAIRS_PATH.read_bytes()).hexdigest() == SUMMARY['paired_arrays_sha256']
assert ENTRY['starts'] == 20 and ENTRY['posthoc'] is True
assert ENTRY['leading_lower_error_starts'] == 20
assert SUMMARY['matched_first_observations_all_arms'] is True

arrays = np.load(PAIRS_PATH, allow_pickle=False)
assert arrays['successes'].shape == (4, 20)
assert arrays['task_ids'].tolist() == [t for t in range(10) for _ in range(2)]
assert arrays['episode_ids'].tolist() == [0, 1] * 10
names = arrays['arm_names'].tolist()
errors = {row['arm']: row['denormalized_rmse_vs_fullrank'] for row in ENTRY['rows']}
outcomes = {name: arrays['successes'][i].tolist() for i, name in enumerate(names)}
for arm in SUMMARY['arms']:
    assert sum(outcomes[arm['arm']]) == arm['successes']
assert outcomes['native'] == outcomes['fullrank']
comparison = next(row for row in SUMMARY['paired_comparisons']
                  if row['a'] == 'anchored_k174_s0' and row['b'] == 'leading_k174')
lead = outcomes['leading_k174']
random = outcomes['anchored_k174_s0']
random_only = sum(a and not b for a, b in zip(random, lead))
lead_only = sum(a and not b for a, b in zip(lead, random))
assert (random_only, lead_only) == (3, 0)
p_exact = min(1.0, 2 * sum(math.comb(3, k) for k in range(min(random_only, lead_only) + 1)) / 2**3)
assert p_exact == comparison['exact_mcnemar_two_sided_p'] == 0.25
ratio = errors['anchored_k174_s0'] / errors['leading_k174']
assert 59.5 < ratio < 59.7

# Sizes are in points. At manuscript width the smallest labels remain 8 pt.
W, H = 720, 252
art = Drawing(W, H)
INK = '#1C2833'
MUTED = '#65727E'
GRID = '#E4E8ED'
ORIGINAL = '#6E7B88'
ODT = '#0072B2'
RANDOM = '#D55E00'
FAIL = '#9AA4AD'
WHITE = '#FFFFFF'


def text(x, y, value, size=14, color=INK, bold=False, align='start'):
    art.add(String(x, y, value, fontName='Helvetica-Bold' if bold else 'Helvetica',
                   fontSize=max(size, 14.6), fillColor=HexColor(color), textAnchor=align))


def line(x1, y1, x2, y2, color=GRID, width=0.8):
    art.add(Line(x1, y1, x2, y2, strokeColor=HexColor(color), strokeWidth=width))


def cross(x, y, size=3.3, color=FAIL, width=1.5):
    line(x-size, y-size, x+size, y+size, color, width)
    line(x-size, y+size, x+size, y-size, color, width)


art.add(Rect(0, 0, W, H, fillColor=HexColor(WHITE), strokeColor=None))
text(8, 236, 'Keep 174 of 193 directions', 12, MUTED)
text(104, 212, 'a   Action change', 18, bold=True)
text(104, 193, 'Initial prediction error', 12, MUTED)
text(411, 212, 'b   Task success', 18, bold=True)
text(411, 193, '20 paired starts', 12, MUTED)
line(377, 46, 377, 217, GRID, 1.1)

rows = [('native', 'Original', ORIGINAL),
        ('leading_k174', 'ODT', ODT),
        ('anchored_k174_s0', 'Random', RANDOM)]
ys = [157, 117, 77]
axis_x, axis_w, axis_y, axis_max = 104, 175, 49, 0.06
for tick in [0, 0.02, 0.04, 0.06]:
    x = axis_x + axis_w*tick/axis_max
    line(x, axis_y, x, 173)
    text(x, axis_y-16, '0' if tick == 0 else f'{tick:.2f}', 11.5, MUTED, align='middle')
line(axis_x, axis_y, axis_x+axis_w, axis_y, '#B5BEC6', 0.9)
for (arm, label, color), y in zip(rows, ys):
    text(8, y-5, label, 16, color, bold=True)
    length = axis_w*errors[arm]/axis_max
    # Exact linear bar length, with no artificial minimum for tiny errors.
    art.add(Rect(axis_x, y-7, length, 14,
                 fillColor=HexColor(color), strokeColor=None))
    display = '<0.000001' if arm == 'native' else f'{errors[arm]:.4f}'
    text(297, y-5, display, 12.5, color, bold=True)
    for j, success in enumerate(outcomes[arm]):
        x = 413 + j*10 + (j//2)*3
        if success:
            art.add(Circle(x, y, 3.8, fillColor=HexColor(color), strokeColor=None))
        else:
            cross(x, y)
    text(699, y-5, f'{sum(outcomes[arm])}/20', 17, color, bold=True, align='end')

# One annotation per panel, with the paired sample visible above it.
text(104, 7, f'{ratio:.0f}× smaller error', 15, ODT, bold=True)
text(411, 7, 'Pilot inconclusive', 15, bold=True)
text(699, 7, f'p = {p_exact:.2f}', 12, MUTED, align='end')
art.add(Circle(415, 38, 3.5, fillColor=HexColor(MUTED), strokeColor=None))
text(425, 34, 'Success', 11.5, MUTED)
cross(490, 38, size=3)
text(501, 34, 'Failure', 11.5, MUTED)

OUTPUT.mkdir(parents=True, exist_ok=True)
PDF.parent.mkdir(parents=True, exist_ok=True)
renderPDF.drawToFile(art, str(PDF), title='Action change and task success', author='')
renderSVG.drawToFile(art, str(OUTPUT / 'odt_actions_success.svg'))
metadata = {
    'source_sha256': {name: hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()
                      for name in ['summary.json', 'entry_descriptive_v1.json', 'paired_episodes.npz']},
    'arms': [{'arm': arm, 'label': label, 'initial_action_rmse': errors[arm],
              'successes': sum(outcomes[arm]), 'episodes': 20,
              'outcomes_in_plot_order': outcomes[arm]} for arm, label, _ in rows],
    'random_over_odt_initial_action_rmse': ratio,
    'exact_paired_p': p_exact,
    'posthoc_initial_actions': True,
    'bar_axis': {'minimum': 0, 'maximum': axis_max, 'scale': 'linear'},
    'scope': 'One checkpoint, one bond at rank 174, one anchored random seed, 20 paired starts.',
    'pdf_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
}
(OUTPUT / 'odt_actions_success.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps({'pdf': str(PDF), 'error_ratio': ratio, 'successes': [sum(outcomes[a]) for a,_,_ in rows]}))
