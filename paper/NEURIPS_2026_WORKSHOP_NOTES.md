# NeurIPS 2026 workshop targeting notes

Research rechecked against the live workshop calls on August 26, 2026.

## Recommended portfolio

Prepare three distinct versions, but do not submit substantially the same work to more than one
NeurIPS workshop concurrently unless every affected workshop chair confirms that this is allowed.
The three target calls below do not publish a clear cross-workshop duplicate-submission rule.
Other NeurIPS 2026 workshops explicitly prohibit concurrent workshop submissions, so silence is not
safe authorization.

Recommended order:

1. Neural Network Artifacts for the strongest current evidence-to-call match.
2. VLM4RWD for the strongest overall topical fit, with the nonzero matched cost stated directly.
3. Robot Learning Workshop for the broadest robotics audience, with a real theme mismatch on
   zero-shot generalization.

## 1. Grounded and Faithful Vision-Language Models for Real-World Deployment

Workshop: [VLM4RWD](https://vlm4rwd.github.io/)

This is the strongest match because its call explicitly requests VLA architectures, reliable
embodied decision-making, causal grounding, counterfactual reasoning, failure-mode analysis, and
interpretability. The paper supplies all of those except a successful grounding repair, and it
reports that boundary directly.

Confirmed requirements:

- Deadline: August 30, 2026. Treat the website date as the hard deadline.
- Format: official NeurIPS 2026 conference format.
- Length: 8 content pages. References and appendices are excluded.
- Review: double blind and fully anonymized.
- Status: non-archival.
- Venue: Sydney, Australia, December 11, 2026.
- Submission: [VLM4RWD OpenReview](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FVLM4RWD).
- Local source: `chi-vla-vlm4rwd.tex`.
- Compiled output: `chi-vla-vlm4rwd.pdf`, 7 content pages with references starting on page 8.
  The complete PDF is 11 pages including references and appendix.

Framing: capable specialist control, exact architectural audit, causal inspection, then the
counterfactual language shortcut as an honest deployment failure.

## 2. Neural Network Artifacts as a New Data Modality

Workshop: [Neural Network Artifacts](https://artifactsasdata.org/cfp/)

This is the strongest alternative for the exact reduced-Gram construction, runtime checkpoint
audit, weight-derived subspaces, model editing, and the informative no-go surgery result. The
workshop explicitly welcomes weights, gradients, representations, interpretability, editing,
pruning, steering, early-stage results, and negative results.

Confirmed requirements:

- Deadline: September 1, 2026, anywhere on Earth.
- Format: NeurIPS 2026 format.
- Extended abstract: 4 to 6 content pages.
- Full paper: 8 to 12 content pages.
- References and supplementary material are excluded from the page limit.
- Venue: Paris and remote participation.
- NeurIPS 2026 workshop guidance states that workshop papers are non-archival. The organizers
  should be asked if either submission track has any additional publication policy.
- Local source: `chi-vla-neuralartifacts.tex`.
- Submission track: full paper.
- Compiled output: `chi-vla-neuralartifacts.pdf`, 9 content pages with references starting on
  page 10. The complete PDF is 12 pages including references and appendix.

Framing: the trained checkpoint is the primary artifact. Capability establishes that the artifact
is behaviorally meaningful. Exact layerwise structure and the causal subspace are the positive
results, while the failed direct edit defines the next research question.

## 3. 8th Robot Learning Workshop

Workshop: [Robot Learning Workshop](https://www.robot-learning.ml/2026/submissions/)

The call directly solicits VLAs, generalist policies, real-world deployment, and failure analysis.
It offers the broadest robotics audience. Its theme asks whether Physical AI is going zero-shot,
while this paper establishes specialist capability and explicitly disclaims zero-shot transfer.
That mismatch lowers the fit relative to the first two targets.

Confirmed requirements:

- Deadline: August 26, 2026, anywhere on Earth.
- Format: NeurIPS 2026 format.
- Length: 6 content pages. References are excluded.
- Review: double blind.
- Status: non-archival.
- Local source: `chi-vla-wrl.tex`.
- Compiled output: `chi-vla-wrl.pdf`, 6 content pages with references starting on page 7.
  The complete PDF is 10 pages including references and appendix.

The current WRL FAQ explicitly allows parallel or later conference and journal submission, but it
does not explicitly authorize sending substantially the same manuscript to another NeurIPS 2026
workshop. The portfolio-level caution above therefore remains.

Framing: specialist Physical AI can be capable and inspectable, but benchmark success conceals a
language shortcut. This directly answers the theme with a boundary rather than claiming zero-shot
generalization.

## Other plausible workshop

[AXIOM](https://axiom-neurips2026.github.io/) fits mathematically structured and interpretable
architectures, but its 4-page limit and emphasis on efficiency make it a weaker choice until the
paper reports matched FLOPs, latency, memory, and energy measurements.

## Format imported into this repository

- `neurips_2026.sty` is unchanged from the official NeurIPS 2026 author kit.
- SHA-256: `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.
- Every version uses the official `dblblindworkshop` option and workshop-specific
  `\workshoptitle` text.
- Manual geometry, caption sizing, and compact-list overrides are absent, so the venue style
  controls layout.
- All abstracts are one paragraph.
- The general VLM audit copy, `chi-vla.pdf`, has 8 content pages with references starting on
  page 9. Its complete PDF is 12 pages including references and appendix.
- The appendix adds per-task ensemble behavior, three-checkpoint and conventional-control
  grounding, systems profiling, intervention diagnostics, and the expanded all-attention plus
  direct-surgery screen.

Compile each source twice from `paper/`:

```bash
pdflatex -interaction=nonstopmode chi-vla-vlm4rwd.tex
pdflatex -interaction=nonstopmode chi-vla-vlm4rwd.tex
pdflatex -interaction=nonstopmode chi-vla-wrl.tex
pdflatex -interaction=nonstopmode chi-vla-wrl.tex
pdflatex -interaction=nonstopmode chi-vla-neuralartifacts.tex
pdflatex -interaction=nonstopmode chi-vla-neuralartifacts.tex
```
