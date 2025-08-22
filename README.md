# unsw-nb15-network-fault-detection:

# Ensemble Learning for Classification with Soft Voting

## Project Overview

This project applies **ensemble learning techniques** to the UNSW-NB15 dataset for network intrusion detection. The goal is to build robust models and compare their performance, ultimately using **soft voting** to combine the best models for improved accuracy and generalization. A key challenge is the severe **class imbalance** in the original data, which was addressed by creating balanced samples for training and testing.

I experimented with:

  - **Random Forest (RF)**
  - **HistGradientBoosting (HGB)**
  - **Soft Voting Classifier** (RF + HGB)

-----

## Dataset

This project uses the **UNSW-NB15 dataset**, a benchmark dataset for network intrusion detection systems. The primary task is binary classification to distinguish between normal network traffic and malicious attacks based on the `label` column.

-----

## Workflow

1.  **Data Preprocessing & Sampling**

      * Created balanced training (8,000 samples) and testing (4,000 samples) sets using **stratified sampling** to handle class imbalance.
      * Applied **one-hot encoding** to categorical features (`proto`, `service`, `state`).
      * Dropped the `attack_cat` column to prevent data leakage.

2.  **Feature Selection**

      * Removed highly redundant features with a **correlation greater than 0.9**.
      * Pruned the **bottom 20% of least important features** using Random Forest's feature importance scores.

3.  **Model Tuning & Training**

      * Tuned hyperparameters for both Random Forest and HistGradientBoosting using **RandomizedSearchCV** with 5-fold cross-validation, optimizing for F1-score.
      * Trained the tuned models on the reduced feature set.
      * Combined the optimized models into a **Soft Voting Ensemble** with equal weights.

4.  **Evaluation**

      * Assessed models using a standard set of metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC.
      * Validation was performed using **5-fold Stratified K-Fold CV** on the training sample.

-----

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.885 | 0.971 | 0.793 | 0.873 | 0.979 |
| HistGradientBoosting | 0.877 | 0.972 | 0.777 | 0.864 | 0.980 |
| **Soft Voting Ensemble**| **0.879** | **0.971** | **0.782** | **0.866** | **0.980** |


 The **Random Forest Classifier** achieved the highest F1-score, making it the top-performing individual model. The **Soft Voting Ensemble** showed comparable performance, offering robust predictions but without a significant boost over the best single model. All top models demonstrated a trade-off between **high precision** (few false alarms) and slightly **lower recall** (some attacks missed).

-----

## Key Takeaways & Conclusion

  * **Ensemble methods are highly effective**: Tree-based ensembles like Random Forest and HistGradientBoosting are well-suited for this intrusion detection task.
  * **Precision is a key strength**: The models are very reliable when flagging an attack (around 97% precision), which is crucial for minimizing false alarms in a real-world cybersecurity context.
  * **Future work could focus on recall**: While precision is high, there's room to improve the detection rate of malicious traffic (around 78% recall). Techniques like threshold tuning or cost-sensitive learning could be explored to enhance recall.

-----

## How to Run

1.  Clone the repository:
    ```bash
    git clone https://your-repo-link.git
    cd unsw-nb15-network-fault-detection
    ```
2.  Install the required dependencies (ensure you have a `requirements.txt` file):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook unsw_model_tournament.ipynb
    ```
