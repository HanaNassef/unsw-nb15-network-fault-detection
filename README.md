# unsw-nb15-network-fault-detection:

# Ensemble Learning for Classification with Soft Voting

## Project Overview

This project applies **ensemble learning techniques** to the UNSW-NB15 dataset for network intrusion detection. The goal is to build robust models and compare their performance, ultimately using **soft voting** to combine the best models for improved accuracy and generalization. A key challenge is the severe **class imbalance** in the original data, which was addressed by creating balanced samples for training and testing.


-----

## Dataset

This project uses the **UNSW-NB15 dataset**, a benchmark dataset for network intrusion detection systems. The primary task is binary classification to distinguish between normal network traffic and malicious attacks based on the `label` column.

-----

## Workflow

1.  **Data Preprocessing & Sampling**
    * Created balanced training (8,000 samples) and testing (4,000 samples) sets using **stratified sampling** to handle the dataset's class imbalance.
    * Applied **one-hot encoding** to categorical features (`proto`, `service`, `state`).
    * Dropped the `attack_cat` column to prevent data leakage during training.

2.  **Baseline Model Tournament**
    * Conducted an initial tournament with five different models: **Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and K-Nearest Neighbors**.
    * Each model was evaluated on the hold-out test set to establish a performance baseline and identify the most promising candidates for further optimization.

3.  **Feature Selection & Optimization**
    * Based on the initial results, **Random Forest** and **HistGradientBoosting** were selected as the top-performing models.
    * To improve these models, highly redundant features with a **correlation greater than 0.9** were removed.
    * The **bottom 20% of least important features** were pruned using Random Forest's feature importance scores.

4.  **Model Tuning & Ensembling**
    * Tuned hyperparameters for both Random Forest and HistGradientBoosting using **RandomizedSearchCV** with 5-fold cross-validation, optimizing for the F1-score.
    * Combined the two optimized models into a **Soft Voting Ensemble** with equal weights to create a more robust final model.

5.  **Evaluation**
    * Assessed all models using a standard set of metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC.
    * Validation was performed using **5-fold Stratified K-Fold CV** on the training sample.

-----

## Results

### Baseline Model Tournament

An initial comparison of five models was performed on the balanced dataset without extensive feature selection or tuning. The results clearly indicated that tree-based ensembles were the most effective.

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.885** | **0.971** | **0.793** | **0.873** | **0.979** |
| **HistGradientBoosting** | 0.877 | 0.972 | 0.777 | 0.864 | 0.980 |
| Decision Tree | 0.869 | 0.949 | 0.780 | 0.856 | 0.869 |
| K-Nearest Neighbors | 0.763 | 0.883 | 0.607 | 0.719 | 0.897 |
| Logistic Regression | 0.710 | 0.882 | 0.484 | 0.625 | 0.859 |

As shown, **Random Forest** and **HistGradientBoosting** were the clear winners with the highest F1 and ROC AUC scores. In contrast, Logistic Regression struggled significantly with recall, and KNN was less performant and slower in prediction. Based on this, the top two models were selected for further optimization.

### Optimized & Ensemble Model Results

After feature selection and hyperparameter tuning, the final models were evaluated.

| Model (Tuned) | Accuracy | Precision | Recall | F1-score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.885 | 0.971 | 0.793 | 0.873 | 0.979 |
| HistGradientBoosting | 0.877 | 0.972 | 0.777 | 0.864 | 0.980 |
| **Soft Voting Ensemble**| **0.879** | **0.971** | **0.782** | **0.866** | **0.980** |

The **tuned Random Forest Classifier** remained the top-performing individual model. The **Soft Voting Ensemble** delivered comparable, robust predictions but did not provide a significant boost over the best single model in this case. All top models demonstrated a trade-off between **high precision** (few false alarms) and slightly **lower recall** (some attacks missed), which can be adjusted depending on the specific use-case.

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
