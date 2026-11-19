# Credit Card Fraud Detection Using Anomaly Detection

An unsupervised machine learning project exploring how **Isolation Forest** and a neural-network **Autoencoder** detect unusual credit card transactions, both **with and without Principal Component Analysis (PCA)**.

## Dataset

**Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains **284,807 transactions**, including **492 fraudulent transactions (approximately 0.173%)**. The target column `Class` identifies normal (`0`) and fraudulent (`1`) transactions. The input columns include `Time`, `Amount`, and anonymized features `V1`–`V28`.

Download `creditcard.csv` from Kaggle and place it at:

```text
data/creditcard.csv
```

The dataset is **not included in this repository** because of its size. The notebook uses `data/creditcard.csv` as its input path.

## Project Workflow

1. Explore class imbalance, transaction amounts and times, and feature correlations.
2. Normalize features using `MinMaxScaler`.
3. Apply PCA to investigate a reduced feature representation (13 components retained in this notebook).
4. Run Isolation Forest on the original scaled features and on PCA features.
5. Train Autoencoders to reconstruct transactions with and without PCA.
6. Flag anomalies using model outputs or high reconstruction error (99th-percentile threshold for the Autoencoders).
7. Evaluate the flagged transactions against `Class` using precision, recall, F1-score, classification reports, and confusion matrices.
8. Explore characteristics of flagged transactions.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow

Install dependencies:

```bash
python -m pip install notebook pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## How to Run

1. Clone this repository.
2. Download the Kaggle dataset and save it as `data/creditcard.csv`.
3. Install the dependencies above.
4. Open the notebook:

   ```bash
   jupyter notebook
   ```

5. Select `Anomaly_Detection_in_Credit_Card_Transactions.ipynb` and run the cells from top to bottom.

The notebook sets `tf.keras.utils.set_random_seed(42)` to help make Autoencoder training reproducible. Exact results can still vary with environment and hardware.

## Results

The following **fraud-class** metrics are taken from the saved notebook outputs:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Isolation Forest | 10.14% | 58.74% | 17.30% |
| **Autoencoder (without PCA)** | **13.72%** | **79.47%** | **23.41%** |
| Isolation Forest + PCA | 12.43% | 71.95% | 21.19% |
| Autoencoder + PCA | 11.97% | 69.31% | 20.41% |

**Best observed result:** The Autoencoder without PCA had the highest precision, recall, and F1-score in this notebook. It identified **391 of the 492 fraudulent transactions**.

PCA improved Isolation Forest's reported F1-score, but reduced the Autoencoder's reported F1-score. All four approaches had relatively low fraud precision, indicating many false-positive alerts.

> **Evaluation limitation:** These metrics are **in-sample**. The notebook trains and evaluates the anomaly detection models using the same full dataset, so the results should **not** be interpreted as performance on unseen transactions. A future version should use a held-out test set, fit preprocessing on training data only, and select thresholds without consulting the test set.

## Future Improvements

- Evaluate on a separate test set to estimate generalization.
- Train the Autoencoder using normal training transactions only.
- Tune anomaly thresholds using a validation set.
- Investigate ways to reduce false positives while maintaining high fraud recall.

## Repository Notes

The notebook expects the large dataset to be downloaded separately. Keep `data/creditcard.csv` out of ordinary Git commits; GitHub rejects files over 100 MB unless an appropriate large-file workflow is used.
