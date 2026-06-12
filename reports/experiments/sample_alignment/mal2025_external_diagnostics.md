# MAL 2025 External Diagnostics

## MAL 2025 overlap label sanity

This check uses the MAL 2025 overlap rows, not the earlier MAL July label-check file.
It verifies whether MAL `members` and `score * 10` are aligned with AniList labels before using MAL-only rows as an external exam.

| source          | target     |     n |   spearman |   pearson |       r2 |   calibration_slope |   calibration_intercept | log_mae            | mae                |
|:----------------|:-----------|------:|-----------:|----------:|---------:|--------------------:|------------------------:|:-------------------|:-------------------|
| mal2025_overlap | popularity | 13740 |   0.983588 |  0.984211 | 0.500929 |            0.935052 |                -0.65726 | 1.2656822289343628 |                    |
| mal2025_overlap | meanScore  | 13740 |   0.944556 |  0.941862 | 0.770409 |            1.16435  |               -14.1897  |                    | 3.9705240174672487 |

## Run22 external calibration summary

Calibration is computed on existing Run22 prediction CSVs. Popularity is evaluated in `log1p` space; meanScore is evaluated in raw 0-100 score space.

| source                                | target     |    n |   spearman |   pearson |        r2 |   calibration_slope |   calibration_intercept | log_mae            | mae                |
|:--------------------------------------|:-----------|-----:|-----------:|----------:|----------:|--------------------:|------------------------:|:-------------------|:-------------------|
| pop_only:no_yolo                      | popularity | 3765 |   0.499791 |  0.548463 |  0.297925 |            1.04354  |               -0.349291 | 1.0358986133099288 |                    |
| dual:no_yolo                          | popularity | 1202 |   0.56472  |  0.514533 | -0.118818 |            0.84643  |                1.92904  | 1.170746284220097  |                    |
| dual:no_yolo                          | meanScore  | 1202 |   0.577041 |  0.558145 | -0.325287 |            0.588989 |               27.6318   |                    | 6.3363338124521995 |
| pop_only:cover_yolo                   | popularity | 3765 |   0.521295 |  0.568865 |  0.317255 |            1.16219  |               -1.0094   | 1.0142722888890516 |                    |
| dual:cover_yolo                       | popularity | 1202 |   0.607284 |  0.533952 | -0.161293 |            0.930968 |                1.44305  | 1.2001040465045525 |                    |
| dual:cover_yolo                       | meanScore  | 1202 |   0.599916 |  0.581707 | -0.357314 |            0.636701 |               25.2865   |                    | 6.491945130387505  |
| pop_only:cover_yolo_coverbanner_proxy | popularity | 3765 |   0.516612 |  0.565251 |  0.314914 |            1.12951  |               -0.831671 | 1.0174095391849345 |                    |
| dual:cover_yolo_coverbanner_proxy     | popularity | 1202 |   0.595459 |  0.529763 | -0.143289 |            0.905679 |                1.58453  | 1.1880000004271338 |                    |
| dual:cover_yolo_coverbanner_proxy     | meanScore  | 1202 |   0.592144 |  0.573702 | -0.337398 |            0.619724 |               26.0981   |                    | 6.40983325329864   |

## Prediction-quantile calibration bins

Rows are grouped by predicted value quantiles. Monotonic actual means indicate useful ranking transfer; systematic gaps between predicted and actual means indicate scale mismatch.

| exam     | variant                      | target     |   pred_quantile |   n |   pred_mean |   actual_mean |   pred_median |   actual_median | mean_log_error       | actual_to_pred_ratio   | mean_error          | mae                |
|:---------|:-----------------------------|:-----------|----------------:|----:|------------:|--------------:|--------------:|----------------:|:---------------------|:-----------------------|:--------------------|:-------------------|
| pop_only | no_yolo                      | popularity |               1 | 753 |    178.792  |      326.579  |      183.826  |          170    | -0.15189197481755862 | 1.8265877911854247     |                     |                    |
| pop_only | no_yolo                      | popularity |               2 | 753 |    312.086  |      444.471  |      308.148  |          192    | 0.28181433842964854  | 1.4241972398172122     |                     |                    |
| pop_only | no_yolo                      | popularity |               3 | 753 |    466.357  |     1571.53   |      462.303  |          270    | 0.3396248761157463   | 3.369800854750597      |                     |                    |
| pop_only | no_yolo                      | popularity |               4 | 753 |    726.459  |     3114.23   |      705.154  |          432    | 0.23495643336005853  | 4.286864576471175      |                     |                    |
| pop_only | no_yolo                      | popularity |               5 | 753 |   2214.47   |    13601.9    |     1516.54   |         2180    | -0.3122419112118463  | 6.142288711056715      |                     |                    |
| dual     | no_yolo                      | popularity |               1 | 241 |    246.07   |     1012.66   |      253.489  |          543    | -1.0528050780616456  | 4.115344166839726      |                     |                    |
| dual     | no_yolo                      | popularity |               2 | 240 |    466.658  |     2751.08   |      466.447  |          735.5  | -0.889787133785142   | 5.895275119911323      |                     |                    |
| dual     | no_yolo                      | popularity |               3 | 240 |    725.77   |     5437.68   |      713.093  |         1066.5  | -0.7788109687997046  | 7.492285005445155      |                     |                    |
| dual     | no_yolo                      | popularity |               4 | 240 |   1161.02   |    10612.5    |     1107.25   |         3153    | -1.123010931751135   | 9.140683922432062      |                     |                    |
| dual     | no_yolo                      | popularity |               5 | 241 |   3463.68   |    22517      |     2600.65   |         4366    | -0.7000827790615117  | 6.500894507551903      |                     |                    |
| dual     | no_yolo                      | meanScore  |               1 | 241 |     45.4475 |       54.8037 |       46.0492 |           54.4  |                      |                        | -9.356228523403475  | 9.544209908319182  |
| dual     | no_yolo                      | meanScore  |               2 | 240 |     52.4213 |       58.0229 |       52.6513 |           57.8  |                      |                        | -5.601588365521719  | 6.1788965556826145 |
| dual     | no_yolo                      | meanScore  |               3 | 240 |     56.3456 |       60.2867 |       56.3538 |           60.15 |                      |                        | -3.9410632643331374 | 5.703597896481248  |
| dual     | no_yolo                      | meanScore  |               4 | 240 |     59.629  |       63.1638 |       59.6008 |           62.7  |                      |                        | -3.5347892433661494 | 5.2760790663228    |
| dual     | no_yolo                      | meanScore  |               5 | 241 |     63.6677 |       65.3307 |       63.0077 |           64.7  |                      |                        | -1.663001450164705  | 4.971207511310466  |
| pop_only | cover_yolo                   | popularity |               1 | 753 |    182.149  |      285.773  |      185.974  |          169    | -0.10845495547810903 | 1.568894181053225      |                     |                    |
| pop_only | cover_yolo                   | popularity |               2 | 753 |    299.88   |      412.79   |      300.493  |          191    | 0.28338179305046807  | 1.3765171457752856     |                     |                    |
| pop_only | cover_yolo                   | popularity |               3 | 753 |    431.82   |     1045.11   |      429.901  |          265    | 0.3202153316902663   | 2.4202500002525222     |                     |                    |
| pop_only | cover_yolo                   | popularity |               4 | 753 |    653.531  |     3086.13   |      636.891  |          395    | 0.14836181581632163  | 4.722244712285701      |                     |                    |
| pop_only | cover_yolo                   | popularity |               5 | 753 |   1893.85   |    14228.9    |     1321.74   |         2614    | -0.5864774587206576  | 7.513225935483724      |                     |                    |
| dual     | cover_yolo                   | popularity |               1 | 241 |    240.072  |      826.386  |      243.383  |          539    | -1.000921111638628   | 3.442242840355925      |                     |                    |
| dual     | cover_yolo                   | popularity |               2 | 240 |    442.634  |     2776.44   |      443.287  |          730    | -0.8783311427874675  | 6.27254495888846       |                     |                    |
| dual     | cover_yolo                   | popularity |               3 | 240 |    669.452  |     4125.97   |      663.242  |         1192    | -0.880992983900555   | 6.1631988363219525     |                     |                    |
| dual     | cover_yolo                   | popularity |               4 | 240 |   1062.41   |    12155.5    |     1034.8    |         3242.5  | -1.300854797361931   | 11.44137308729815      |                     |                    |
| dual     | cover_yolo                   | popularity |               5 | 241 |   2891.45   |    22447.7    |     2189.24   |         4519    | -0.889555058089771   | 7.76349414328265       |                     |                    |
| dual     | cover_yolo                   | meanScore  |               1 | 241 |     45.3719 |       54.3701 |       46.019  |           54.2  |                      |                        | -8.998232647446292  | 9.18013750608097   |
| dual     | cover_yolo                   | meanScore  |               2 | 240 |     52.0041 |       58.015  |       52.1377 |           57.8  |                      |                        | -6.0109393261890425 | 6.565156915223436  |
| dual     | cover_yolo                   | meanScore  |               3 | 240 |     55.7986 |       60.6679 |       55.9678 |           60.35 |                      |                        | -4.869341839546075  | 6.066160993639724  |
| dual     | cover_yolo                   | meanScore  |               4 | 240 |     58.9919 |       63.1392 |       58.9268 |           62.95 |                      |                        | -4.147316146404497  | 5.686518931627747  |
| dual     | cover_yolo                   | meanScore  |               5 | 241 |     62.9668 |       65.417  |       62.4348 |           64.7  |                      |                        | -2.4502410435272557 | 4.956946332126349  |
| pop_only | cover_yolo_coverbanner_proxy | popularity |               1 | 753 |    181.446  |      326.38   |      184.68   |          171    | -0.13705814547385295 | 1.7987700674728169     |                     |                    |
| pop_only | cover_yolo_coverbanner_proxy | popularity |               2 | 753 |    303.677  |      412.696  |      302.337  |          189    | 0.29751002604976096  | 1.3589953978875289     |                     |                    |
| pop_only | cover_yolo_coverbanner_proxy | popularity |               3 | 753 |    441.805  |     1352      |      439.876  |          267    | 0.329504839503123    | 3.060164786992018      |                     |                    |
| pop_only | cover_yolo_coverbanner_proxy | popularity |               4 | 753 |    672.932  |     3070.25   |      654.466  |          394    | 0.17711651433488876  | 4.562505720532207      |                     |                    |
| pop_only | cover_yolo_coverbanner_proxy | popularity |               5 | 753 |   1992.5    |    13897.4    |     1385.24   |         2499    | -0.5067761480249445  | 6.974873511146758      |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | popularity |               1 | 241 |    242.388  |      869.751  |      243.253  |          539    | -1.0065066239247142  | 3.5882648055027575     |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | popularity |               2 | 240 |    451.308  |     2787.29   |      452.276  |          731.5  | -0.8717095561660708  | 6.176021146178566      |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | popularity |               3 | 240 |    683.634  |     4936.15   |      678.794  |         1192    | -0.8931946207163205  | 7.220456019656724      |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | popularity |               4 | 240 |   1092.62   |    11509.7    |     1061.37   |         3242.5  | -1.241689364337911   | 10.53405887033321      |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | popularity |               5 | 241 |   3076.8    |    22229.8    |     2327.22   |         4402    | -0.8026418867269729  | 7.224978661017487      |                     |                    |
| dual     | cover_yolo_coverbanner_proxy | meanScore  |               1 | 241 |     45.4255 |       54.5477 |       45.9756 |           54.3  |                      |                        | -9.122233025351797  | 9.313250028952291  |
| dual     | cover_yolo_coverbanner_proxy | meanScore  |               2 | 240 |     52.178  |       58.0483 |       52.4471 |           57.8  |                      |                        | -5.870346485249177  | 6.385826210754516  |
| dual     | cover_yolo_coverbanner_proxy | meanScore  |               3 | 240 |     56.0227 |       60.4288 |       56.1029 |           60.1  |                      |                        | -4.406032693244504  | 5.902337992500468  |
| dual     | cover_yolo_coverbanner_proxy | meanScore  |               4 | 240 |     59.2526 |       63.1754 |       59.1833 |           62.9  |                      |                        | -3.922770838018396  | 5.475877623211235  |
| dual     | cover_yolo_coverbanner_proxy | meanScore  |               5 | 241 |     63.2436 |       65.4083 |       62.685  |           64.7  |                      |                        | -2.1646727974535898 | 4.965793672761701  |
