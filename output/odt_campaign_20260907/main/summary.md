**ODT real-input campaign: paired confirmation results**

Completed 25/25 frozen conditions. One trained checkpoint, real inputs from its training-data source, two-token operator.

Each RMSE uses the condition's jointly valid rows with the source. Invalidity uses every frozen input.

| ID | Allocation / removed dimensions | Retained-space construction | Real RMSE | Real invalid | Gaussian RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | full K=0 | leading | 2.20629e-14 | 0/320 | 1.15092e-14 |
| 1 | W K=93, WN K=93 | leading | 0.0420297 | 0/320 | 0.058148 |
| 2 | W K=464, WN K=464 | leading | 0.0868945 | 0/320 | 0.106534 |
| 3 | W K=927, WN K=927 | leading | 0.109973 | 0/320 | 0.142223 |
| 4 | W K=927, WN K=927 | zero-input anchored QR, seed 0 | 1.06455 | 0/320 | 0.573082 |
| 5 | W K=927, WN K=927 | zero-input anchored QR, seed 1 | 1.06063 | 0/320 | 0.575477 |
| 6 | W K=927, WN K=927 | zero-input anchored QR, seed 2 | 0.952759 | 0/320 | 0.549312 |
| 7 | W K=2782 | leading | 0.374901 | 0/320 | 0.467858 |
| 8 | W K=2782 | zero-input anchored QR, seed 0 | 537.354 | 0/320 | 3.99783 |
| 9 | W K=2782 | zero-input anchored QR, seed 1 | 28.0116 | 0/320 | 4.19011 |
| 10 | W K=2782 | zero-input anchored QR, seed 2 | 60.515 | 0/320 | 3.68836 |
| 11 | WN K=2782 | leading | 0.327004 | 0/320 | 0.334772 |
| 12 | WN K=2782 | zero-input anchored QR, seed 0 | 1.48559 | 0/320 | 0.742835 |
| 13 | WN K=2782 | zero-input anchored QR, seed 1 | 1.24476 | 0/320 | 0.731174 |
| 14 | WN K=2782 | zero-input anchored QR, seed 2 | 1.29769 | 0/320 | 0.730426 |
| 15 | S K=93, SN K=93 | leading | 0.000158501 | 0/320 | 0.000163642 |
| 16 | S K=464, SN K=464 | leading | 0.0314705 | 0/320 | 0.0309649 |
| 17 | S K=927, SN K=927 | leading | 0.0964752 | 0/320 | 0.127134 |
| 18 | S K=927, SN K=927 | zero-input anchored QR, seed 0 | 1.81326 | 0/320 | 0.670609 |
| 19 | S K=927, SN K=927 | zero-input anchored QR, seed 1 | 1.03837 | 0/320 | 0.657134 |
| 20 | S K=927, SN K=927 | zero-input anchored QR, seed 2 | 1.18919 | 0/320 | 0.651609 |
| 21 | S K=2782, SN K=2782 | leading | 0.569509 | 0/320 | 0.333199 |
| 22 | S K=2782, SN K=2782 | zero-input anchored QR, seed 0 | 3.22148 | 0/320 | 0.678179 |
| 23 | S K=2782, SN K=2782 | zero-input anchored QR, seed 1 | 1.68165 | 0/320 | 0.676258 |
| 24 | S K=2782, SN K=2782 | zero-input anchored QR, seed 2 | 1.7473 | 0/320 | 0.676456 |

Paired real-input contrasts use a task-stratified episode bootstrap. Negative squared-error differences favor the first condition. Invalidity is reported separately. Intervals are descriptive and are not adjusted for multiple comparisons.

| Contrast | Jointly valid | Paired MSE difference | 95% interval | Invalidity difference |
| --- | ---: | ---: | --- | ---: |
| S-versus-W-K93 | 320 | -0.00176647 | [-0.0017837, -0.00175067] | 0 |
| SN-versus-WN-K93 | 320 | -0.00176647 | [-0.0017837, -0.00175067] | 0 |
| WN-versus-W-K93 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K93 | 320 | 0 | [0, 0] | 0 |
| S-versus-W-K464 | 320 | -0.00656026 | [-0.00667413, -0.00644219] | 0 |
| SN-versus-WN-K464 | 320 | -0.00656026 | [-0.00667413, -0.00644219] | 0 |
| WN-versus-W-K464 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K464 | 320 | 0 | [0, 0] | 0 |
| S-versus-W-K927 | 320 | -0.00278649 | [-0.00289235, -0.00266865] | 0 |
| SN-versus-WN-K927 | 320 | -0.00278649 | [-0.00289235, -0.00266865] | 0 |
| WN-versus-W-K927 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K927 | 320 | 0 | [0, 0] | 0 |
| W-leading-versus-anchored0-K927 | 320 | -1.12117 | [-1.13325, -1.11081] | 0 |
| W-leading-versus-anchored1-K927 | 320 | -1.11285 | [-1.12317, -1.10341] | 0 |
| W-leading-versus-anchored2-K927 | 320 | -0.895655 | [-0.906768, -0.885388] | 0 |
| WN-leading-versus-anchored0-K927 | 320 | -1.12117 | [-1.13325, -1.11081] | 0 |
| WN-leading-versus-anchored1-K927 | 320 | -1.11285 | [-1.12317, -1.10341] | 0 |
| WN-leading-versus-anchored2-K927 | 320 | -0.895655 | [-0.906768, -0.885388] | 0 |
| S-leading-versus-anchored0-K927 | 320 | -3.2786 | [-3.30848, -3.24802] | 0 |
| S-leading-versus-anchored1-K927 | 320 | -1.06891 | [-1.0873, -1.05315] | 0 |
| S-leading-versus-anchored2-K927 | 320 | -1.40486 | [-1.42401, -1.38755] | 0 |
| SN-leading-versus-anchored0-K927 | 320 | -3.2786 | [-3.30848, -3.24802] | 0 |
| SN-leading-versus-anchored1-K927 | 320 | -1.06891 | [-1.0873, -1.05315] | 0 |
| SN-leading-versus-anchored2-K927 | 320 | -1.40486 | [-1.42401, -1.38755] | 0 |
| S-versus-W-K2782 | 320 | 0.18379 | [0.179962, 0.187371] | 0 |
| SN-versus-WN-K2782 | 320 | 0.217409 | [0.213214, 0.221911] | 0 |
| WN-versus-W-K2782 | 320 | -0.0336193 | [-0.0385069, -0.029905] | 0 |
| SN-versus-S-K2782 | 320 | 0 | [0, 0] | 0 |
| W-leading-versus-anchored0-K2782 | 320 | -288750 | [-524728, -110576] | 0 |
| W-leading-versus-anchored1-K2782 | 320 | -784.509 | [-900.935, -695.024] | 0 |
| W-leading-versus-anchored2-K2782 | 320 | -3661.92 | [-4938.31, -2722.53] | 0 |
| WN-leading-versus-anchored0-K2782 | 320 | -2.10006 | [-2.12797, -2.07438] | 0 |
| WN-leading-versus-anchored1-K2782 | 320 | -1.44249 | [-1.47256, -1.4176] | 0 |
| WN-leading-versus-anchored2-K2782 | 320 | -1.57708 | [-1.60211, -1.55539] | 0 |
| S-leading-versus-anchored0-K2782 | 320 | -10.0536 | [-12.8851, -7.69593] | 0 |
| S-leading-versus-anchored1-K2782 | 320 | -2.50359 | [-2.58431, -2.41809] | 0 |
| S-leading-versus-anchored2-K2782 | 320 | -2.72871 | [-2.76312, -2.69577] | 0 |
| SN-leading-versus-anchored0-K2782 | 320 | -10.0536 | [-12.8851, -7.69593] | 0 |
| SN-leading-versus-anchored1-K2782 | 320 | -2.50359 | [-2.58431, -2.41809] | 0 |
| SN-leading-versus-anchored2-K2782 | 320 | -2.72871 | [-2.76312, -2.69577] | 0 |

Random-control seed summaries describe the three retained-basis seeds. They do not measure policy-training-seed uncertainty. RMSE means summarize each seed's conditional endpoint.

| Allocation | Available basis seeds | Mean real RMSE | Real RMSE range | Mean real invalidity |
| --- | --- | ---: | --- | ---: |
| S K=927, SN K=927 | [0, 1, 2] | 1.34694 | [1.03837, 1.81326] | 0 |
| W K=927, WN K=927 | [0, 1, 2] | 1.02598 | [0.952759, 1.06455] | 0 |
| W K=2782 | [0, 1, 2] | 208.627 | [28.0116, 537.354] | 0 |
| S K=2782, SN K=2782 | [0, 1, 2] | 2.21681 | [1.68165, 3.22148] | 0 |
| WN K=2782 | [0, 1, 2] | 1.34268 | [1.24476, 1.48559] | 0 |
