**ODT real-input campaign: paired confirmation results**

Completed 7/7 frozen conditions. One trained checkpoint, real inputs from its training-data source, two-token operator.

Each RMSE uses the condition's jointly valid rows with the source. Invalidity uses every frozen input.

| ID | Allocation / removed dimensions | Retained-space construction | Real RMSE | Real invalid | Gaussian RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | full K=0 | leading | 2.20629e-14 | 0/320 | 1.15092e-14 |
| 1 | W K=1, WN K=1 | leading | 0.0100496 | 0/320 | 0.00464497 |
| 2 | W K=9, WN K=9 | leading | 0.0181901 | 0/320 | 0.0203346 |
| 3 | W K=46, WN K=46 | leading | 0.0298968 | 0/320 | 0.0398503 |
| 4 | S K=1, SN K=1 | leading | 8.68362e-08 | 0/320 | 8.3773e-08 |
| 5 | S K=9, SN K=9 | leading | 4.60421e-06 | 0/320 | 5.19541e-06 |
| 6 | S K=46, SN K=46 | leading | 5.36027e-05 | 0/320 | 5.84688e-05 |

Paired real-input contrasts use a task-stratified episode bootstrap. Negative squared-error differences favor the first condition. Invalidity is reported separately. Intervals are descriptive and are not adjusted for multiple comparisons.

| Contrast | Jointly valid | Paired MSE difference | 95% interval | Invalidity difference |
| --- | ---: | ---: | --- | ---: |
| S-versus-W-K1 | 320 | -0.000100995 | [-0.000104258, -9.80121e-05] | 0 |
| SN-versus-WN-K1 | 320 | -0.000100995 | [-0.000104258, -9.80121e-05] | 0 |
| WN-versus-W-K1 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K1 | 320 | 0 | [0, 0] | 0 |
| S-versus-W-K9 | 320 | -0.00033088 | [-0.000337463, -0.000324724] | 0 |
| SN-versus-WN-K9 | 320 | -0.00033088 | [-0.000337463, -0.000324724] | 0 |
| WN-versus-W-K9 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K9 | 320 | 0 | [0, 0] | 0 |
| S-versus-W-K46 | 320 | -0.000893813 | [-0.00090882, -0.00087974] | 0 |
| SN-versus-WN-K46 | 320 | -0.000893813 | [-0.00090882, -0.00087974] | 0 |
| WN-versus-W-K46 | 320 | 0 | [0, 0] | 0 |
| SN-versus-S-K46 | 320 | 0 | [0, 0] | 0 |

Random-control seed summaries describe the three retained-basis seeds. They do not measure policy-training-seed uncertainty. RMSE means summarize each seed's conditional endpoint.

| Allocation | Available basis seeds | Mean real RMSE | Real RMSE range | Mean real invalidity |
| --- | --- | ---: | --- | ---: |
