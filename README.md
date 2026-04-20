# CI Build Failure Prediction on TravisTorrent

Wilson Neira

## Overview

This project studies whether a Transformer-based sequential model can improve CI build failure prediction on TravisTorrent-style data compared with strong baselines. The comparison is built around a fair **common10** benchmark so the main models are evaluated on the same 10-project subset. In the current project, the final comparison includes:

- **Parrot_common10**
- **CI_Build_Transformer_common10**
- **SharedSeq_LSTM_common10**
- **BuildFast_common10**
- **DL-CIBuild_common10**

In the latest final results saved by the notebook, **Parrot_common10** is the strongest overall model, while **CI_Build_Transformer_common10** is the strongest learned sequence model. The final saved table shows:

- Parrot_common10: **F1-fail = 0.972487**
- CI_Build_Transformer_common10: **F1-fail = 0.955377**
- SharedSeq_LSTM_common10: **F1-fail = 0.899628**
- BuildFast_common10: **F1-fail = 0.894571**
- DL-CIBuild_common10: **F1-fail = 0.467168**

## Main Idea

The goal is not just to train a Transformer, but to check whether attention-based sequence modeling is strong enough to justify using it against simpler and stronger baselines like Parrot, BuildFast, and DL-CIBuild.

A key point in this project is that **Parrot is a very strong baseline**. It predicts the next build outcome by copying the previous build outcome. On this dataset, build outcomes often repeat, so Parrot is hard to beat. Because of that, the project focuses on building a fair comparison and understanding where the Transformer still helps rather than pretending the Transformer automatically wins.

## What This Project Does

This project:

1. Loads TravisTorrent-style CI build history.
2. Builds a binary label for **fail vs pass**.
3. Sorts builds in time order to avoid leakage.
4. Creates history-aware tabular features.
5. Creates sequence windows for LSTM and Transformer models.
6. Runs a fair **common10** benchmark.
7. Compares Parrot, BuildFast, SharedSeq LSTM, DL-CIBuild, and Transformer rows.
8. Evaluates models with:
   - failure precision / recall / F1
   - accuracy
   - ROC-AUC
   - PR-AUC
   - cost-benefit style metrics such as benefit, cost, and gain in build hours

## Repository / Project Structure

Typical project structure used by the notebook:

```text
CIBuild-Transformer/
├── ci_build_failure_prediction_starter.ipynb
├── ci_build_failure_prediction_starter.html
├── data/
│   ├── travistorrent/
│   ├── buildfast/
│   └── dl_cibuild/
├── BuildFastinCI.github.io-master/
├── DL-CIBuild-main/
├── outputs/
└── models/
```

## Datasets Used

### 1. TravisTorrent
This is the main dataset for the project and the main one the proposal centers on. It provides CI build history, outcomes, and metadata needed for the prediction task.

### 2. BuildFast data / code
Used to produce a stronger literature-based tabular baseline and import a BuildFast-style result into the final comparison.

### 3. DL-CIBuild data / code
Used to reproduce a paper-style deep learning baseline on the same common10 project subset.

## Models Compared

### Parrot_common10
A simple but very strong baseline that predicts the next build outcome by repeating the previous build outcome.

### BuildFast_common10
A BuildFast-style/common10 tabular baseline based on the BuildFast paper and external project code, normalized into the final comparison table.

### SharedSeq_LSTM_common10
A shared-sequence LSTM baseline built from the same fair common10 sequence setup as the Transformer, so the learned sequence models are compared more fairly.

### DL-CIBuild_common10
A DL-CIBuild-based baseline produced from the external DL-CIBuild code and summarized into the final results.

### CI_Build_Transformer_common10
The main project model. This is the Transformer-based sequence model evaluated on the common10 benchmark. In the latest final results, it is the strongest learned sequence model, but it still does not beat Parrot overall.

## Current Best Final Results

From the latest saved notebook output:

| Model | F1-fail | Projects Processed |
|---|---:|---:|
| Parrot_common10 | 0.972487 | 10 |
| CI_Build_Transformer_common10 | 0.955377 | 10 |
| SharedSeq_LSTM_common10 | 0.899628 | 10 |
| BuildFast_common10 | 0.894571 | 10 |
| DL-CIBuild_common10 | 0.467168 | 10 |

## Key Takeaways

- **Parrot is the strongest overall baseline** on the fair common10 benchmark.
- **The Transformer is the strongest learned sequence model** in the current project.
- **SharedSeq LSTM beats BuildFast** in the latest saved result table, but both remain below the Transformer.
- **DL-CIBuild performs much worse** than the other baselines in the current adapted common10 setup.

## How to Run

This project has evolved over time, but the main workflow is:

1. Place the TravisTorrent file in `data/travistorrent/`.
2. Keep BuildFast resources in `BuildFastinCI.github.io-master/` and relevant data in `data/buildfast/`.
3. Keep DL-CIBuild resources in `DL-CIBuild-main/` and relevant data in `data/dl_cibuild/`.
4. Run the notebook `ci_build_failure_prediction_starter.ipynb`.
5. Check the generated outputs in `outputs/`, especially:
   - `model_results.csv`
   - `buildfast_original_summary.csv`
   - `dl_cibuild_original_summary.csv`

## Outputs

Important outputs include:

- `outputs/model_results.csv`
- `outputs/buildfast_original_summary.csv`
- `outputs/buildfast_original_per_project.csv`
- `outputs/dl_cibuild_original_summary.csv`
- `outputs/dl_cibuild_original_per_fold.csv`

## Current Limitations

- The Transformer still does **not** beat Parrot on this benchmark.
- Results are sensitive to preprocessing and benchmark setup, so keeping the comparison fair is important.
- DL-CIBuild is included as a literature-based baseline, but in this current common10 adaptation it performs much worse than expected from the original paper-based motivation.

## Future Work

Planned next steps include:

- improving the Transformer on harder “flip” cases where the next build outcome changes
- running more controlled ablations on sequence features and Transformer settings
- further tightening the fairness of the comparison while trying to improve learned-model performance

## Notes

This README reflects the current state of the project shown in the attached notebook and HTML outputs, not just the older starter version.
