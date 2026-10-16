# fast-VUS
Exact, vectorised VUS-PR in one NumPy file. Same values as the reference implementation (up to floating-point rounding), and 84–939× faster!

For my master thesis I am running many experiments on time series anomaly detection using VUS-PR as the evaluation metric. It is great, since it resolves many issues with other evaluation metrics in that field. It is parameter-free, threshold-independent, robust to label noise, lag between scores and labels and the ratio of anomalous to normal points. That comes at the price of looping over many buffer lengths, score thresholds and anomaly segments. 

I discovered that the reference implementation can be easily vectorized. The result is all the benefits of VUS-PR robustness without most of the extra computational cost. Enjoy!

Currently, only VUS-PR is implemented. I am planning to add VUS-ROC in the future.

This is only a re-implementation. All credit for the metric goes to Paparrizos, Boniol et al. for their wonderful work in the field.

## Install and use

```bash
pip install git+https://github.com/tomlouisklein/fast-VUS.git          # NumPy only
pip install "fast-vus[tsb] @ git+https://github.com/tomlouisklein/fast-VUS.git"  # + vus_pr_auto
```

```python
from fast_vus import vus_pr, vus_pr_auto

vus_pr(label, score, window)       # you pick the window
vus_pr_auto(label, score, series)  # window from TSB-AD's find_length_rank
```

The argument order is `(label, score)`, as in TSB-AD's `generate_curve`. TSB-AD's `get_metrics`
takes `(score, labels)`. `window` is the largest buffer length, usually the period of the series.
`vus_pr_auto` estimates it with TSB-AD's `find_length_rank` (125 if no period is found) and needs
the `[tsb]` extra.

## What VUS-PR measures

VUS-PR (Paparrizos et al., PVLDB 2022) is the volume under the range-based precision-recall
surface: the mean of the range-based AUC-PR over all thresholds and all buffer lengths from 0 to
`window`. 

## Implementation details

For each window buffer length, the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/evaluation/basic_metrics.py) implementation of VUS-PR (`RangeAUC_volume_opt`) loops over 250 anomaly score thresholds, and for each threshold over all anomaly segments. This threshold loop is the bottleneck.

I noticed that TSB-AD computes AUC-PR with [scikit-learn's `average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), while its VUS-PR is a basic for-loop over thresholds and segments.

The [scikit-learn implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py) (`_binary_clf_curve`) uses a simple trick to vectorize this:
1. Sort the points by score, highest first.
2. Take a cumulative sum ([`np.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)) over that order. It holds the sum for every threshold at once.
3. Read one entry per threshold. [`np.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html) gives the position.

This implementation does the same for VUS-PR.

Note that the innovation of VUS over AUC, the loop over all window buffer lengths (creating a surface) remains unchanged.

`vus_pr(label, score, window)` equals TSB-AD's `generate_curve(label, score, window, 'opt', 250)[-1]`.

## Comparison

`compare.py` runs Sub_PCA (TSB-AD defaults) on TSB-AD-U series and computes VUS-PR both ways.

```bash
pip install -r requirements.txt
curl -O https://www.thedatum.org/datasets/TSB-AD-U.zip && unzip TSB-AD-U.zip
python compare.py TSB-AD-U -n 20
```

`--files` takes a TSB-AD file list (e.g. `File_List/TSB-AD-U-Eva.csv`). Results go to `results.csv`.

| series | n | window | TSB-AD | fast | speedup |
|---|---:|---:|---:|---:|---:|
| 149_Stock_id_1 | 20 000 | 112 | 25.65 s | 0.065 s | 393× |
| 158_Stock_id_10 | 20 000 | 42 | 20.37 s | 0.022 s | 939× |
| 223_LTDB_id_8 | 100 000 | 114 | 19.83 s | 0.082 s | 242× |
| 267_IOPS_id_8 | 6 138 | 282 | 2.83 s | 0.032 s | 89× |
| 006_NAB_id_6 | 10 319 | 125 | 1.24 s | 0.015 s | 84× |

VUS-PR cost grows with the window and the number of anomaly events.

Single run, one thread (`OPENBLAS_NUM_THREADS=1`), on a MacBook Air (Apple M5, 4 performance +
6 efficiency cores, 32 GB), macOS 26.6, Python 3.12.15, NumPy 1.26.4, pandas 2.3.3, TSB-AD 1.5.

### Differences to the reference

The formulas are the same, but the additions run in a different order (a cumulative sum over the
score-sorted points instead of per-segment dot products, `np.mean` instead of `sum / len`), so the
results differ by floating-point rounding only: at most 1.4e-15 on 18 TSB-AD-U series.

## Citation

If you use VUS-PR, cite the VUS paper:

```bibtex
@article{paparrizos2022volume,
  title={{Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection}},
  author={Paparrizos, John and Boniol, Paul and Palpanas, Themis and Tsay, Ruey S and Elmore, Aaron and Franklin, Michael J},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={2774--2787},
  year={2022},
  publisher={VLDB Endowment}
}
```

```bibtex
@article{boniol2025vus,
  title={{VUS: Effective and Efficient Accuracy Measures for Time-Series Anomaly Detection}},
  author={Boniol, Paul and Krishna, Ashwin K. and Bruel, Marine and Liu, Qinghua and Huang, Mingyi and Palpanas, Themis and Tsay, Ruey S. and Elmore, Aaron and Franklin, Michael J. and Paparrizos, John},
  journal={The VLDB Journal},
  volume={34},
  pages={32},
  year={2025},
  doi={10.1007/s00778-025-00907-x}
}
```

Code and further measures: [github.com/TheDatumOrg/VUS](https://github.com/TheDatumOrg/VUS).

## License

Apache-2.0. The metric is by Paparrizos et al. (2022). `fast_vus.py` is an independent implementation and copies no code from the VUS or TSB-AD repositories. `vus_pr_auto` imports TSB-AD's `find_length_rank` at runtime.