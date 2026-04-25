# Forecast Evaluation

- Baseline forecaster: `ema`
- Improved forecaster: `adaptive_ema`
- Scored metrics: `MAE`, `RMSE`, `MAPE`
- MAPE ignores zero-demand targets to avoid undefined divisions.

## Aggregate Summary

```text
scenario   forecaster  series_count  points  horizon_steps     mae    rmse  mape_pct  mape_points  mean_actual_demand  mean_predicted_demand  baseline_mae  baseline_rmse  baseline_mape_pct  mae_improvement_pct_vs_baseline  rmse_improvement_pct_vs_baseline  mape_improvement_pct_vs_baseline
     all adaptive_ema             5    4455              3 42.8546 63.6828  106.2588         4446             78.8456                78.4753       44.3483        66.2276           105.6248                           3.3682                            3.8426                           -0.6002
     all          ema             5    4455              3 44.3483 66.2276  105.6248         4446             78.8456                78.7026       44.3483        66.2276           105.6248                           0.0000                            0.0000                            0.0000
   burst adaptive_ema             5    1485              3 60.1506 86.7915   98.6543         1485            113.9717               113.5159       61.4464        89.5752            95.7205                           2.1089                            3.1077                           -3.0650
   burst          ema             5    1485              3 61.4464 89.5752   95.7205         1485            113.9717               113.8298       61.4464        89.5752            95.7205                           0.0000                            0.0000                            0.0000
 hotspot adaptive_ema             5    1485              3 36.0291 51.5899  111.8123         1477             63.7785                63.3481       37.4842        53.8877           111.9908                           3.8820                            4.2641                            0.1594
 hotspot          ema             5    1485              3 37.4842 53.8877  111.9908         1477             63.7785                63.5529       37.4842        53.8877           111.9908                           0.0000                            0.0000                            0.0000
  normal adaptive_ema             5    1485              3 32.3842 44.4095  108.3409         1484             58.7865                58.5620       34.1144        47.2303           109.1998                           5.0720                            5.9723                            0.7865
  normal          ema             5    1485              3 34.1144 47.2303  109.1998         1484             58.7865                58.7252       34.1144        47.2303           109.1998                           0.0000                            0.0000                            0.0000
```

## Per-Series Mean Metrics

```text
scenario  episode_seed   forecaster  horizon_steps  points     mae     rmse  mape_pct  mape_points  mean_actual_demand  mean_predicted_demand
   burst             0 adaptive_ema              3     297 53.3870  77.8006   81.2847          297            106.1313               105.8649
   burst             0          ema              3     297 53.8550  79.3037   76.8995          297            106.1313               106.2260
   burst             1 adaptive_ema              3     297 78.3719 109.2944  129.1070          297            126.0101               124.9795
   burst             1          ema              3     297 81.9600 115.2279  127.6604          297            126.0101               125.6025
   burst             2 adaptive_ema              3     297 59.0252  83.4979   98.2648          297            127.5051               127.5656
   burst             2          ema              3     297 59.7990  85.2808   93.9694          297            127.5051               127.8644
   burst             3 adaptive_ema              3     297 46.0720  63.3457   81.3839          297             97.2424                96.5591
   burst             3          ema              3     297 47.4159  65.7488   80.7145          297             97.2424                96.6998
   burst             4 adaptive_ema              3     297 63.8966  93.1720  103.2309          297            112.9697               112.6103
   burst             4          ema              3     297 64.2019  94.6376   99.3587          297            112.9697               112.7564
 hotspot             0 adaptive_ema              3     297 39.2918  57.6870  103.5524          297             74.0875                73.6311
 hotspot             0          ema              3     297 40.5839  59.3418  105.0834          297             74.0875                73.7839
 hotspot             1 adaptive_ema              3     297 38.7580  53.9150  141.6857          293             62.5488                62.1910
 hotspot             1          ema              3     297 41.1828  57.0136  147.7727          293             62.5488                62.4129
 hotspot             2 adaptive_ema              3     297 38.2592  54.7531  116.9631          293             65.4377                65.0090
 hotspot             2          ema              3     297 39.6424  57.0560  114.8359          293             65.4377                65.2662
 hotspot             3 adaptive_ema              3     297 24.3427  32.6090   75.0880          297             50.0640                49.2082
 hotspot             3          ema              3     297 25.2475  34.2272   75.2195          297             50.0640                49.4719
 hotspot             4 adaptive_ema              3     297 39.4937  54.8792  122.2443          297             66.7542                66.7011
 hotspot             4          ema              3     297 40.7644  57.6242  117.5628          297             66.7542                66.8295
  normal             0 adaptive_ema              3     297 28.7464  37.2199   86.2670          297             57.5960                57.7543
  normal             0          ema              3     297 30.6410  39.3880   90.4197          297             57.5960                57.9004
  normal             1 adaptive_ema              3     297 34.9274  48.9910  125.5094          297             59.3434                58.9479
  normal             1          ema              3     297 37.1886  52.2715  124.2555          297             59.3434                59.1392
  normal             2 adaptive_ema              3     297 31.4549  45.5373  118.5234          297             53.7710                53.2834
  normal             2          ema              3     297 33.0089  48.6489  120.5225          297             53.7710                53.4018
  normal             3 adaptive_ema              3     297 33.5367  43.7405  113.8088          296             57.5657                57.2652
  normal             3          ema              3     297 35.1920  46.3349  112.1740          296             57.5657                57.5080
  normal             4 adaptive_ema              3     297 33.2554  45.7026   97.6146          297             65.6566                65.5592
  normal             4          ema              3     297 34.5417  48.5401   98.6373          297             65.6566                65.6764
```
