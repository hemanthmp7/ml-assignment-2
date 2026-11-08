# Electrical Grid Stability Classification

BITS Pilani M.Tech (DSE) - Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to build, evaluate, and compare 5 distinct classification models to predict the stability of a 4-node smart electric power grid system based on local reaction times, nominal power, and elasticity coefficients. The best-performing models are serialized and integrated into an interactive web application deployed on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset**: UCI Electrical Grid Stability Simulated Data (`Data_for_UCI_named.csv`)
- **Instances**: 10,000 total instances (8,000 Train / 2,000 Test)
- **Features (12 continuous features)**:
  - `tau1` to `tau4`: Reaction time of each network participant
  - `p1` to `p4`: Nominal power produced (positive) or consumed (negative)
  - `g1` to `g4`: Price elasticity coefficient for each network participant
- **Target Variable**: `stabf` — Binary classification (`stable` vs `unstable`). Note: Continuous variable `stab` was removed to prevent target leakage.
- **Split**: 80% Train, 20% Test (Stratified)

## c. GitHub Repository Link
- Replace with your GitHub Repository Link

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8200 | 0.8920 | 0.8408 | 0.8856 | 0.8626 | 0.6039 |
| **Decision Tree** | 0.8330 | 0.8867 | 0.8336 | 0.9224 | 0.8757 | 0.6306 |
| **KNN** | 0.8680 | 0.9374 | 0.8543 | 0.9561 | 0.9024 | 0.7113 |
| **Naive Bayes** | 0.8445 | 0.9210 | 0.8395 | 0.9350 | 0.8847 | 0.6570 |
| **Support Vector Machine** | **0.9635** | **0.9949** | **0.9666** | **0.9765** | **0.9715** | **0.9208** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Acts as an effective linear baseline (Accuracy: 0.8200, AUC: 0.8920). While it handles standardized features smoothly, its lower MCC (0.6039) highlights its limitation in capturing non-linear cross-interactions between price elasticity and grid power. |
| **Decision Tree** | Captures feature interaction thresholds directly without requiring scaling, achieving strong recall (0.9224). However, orthogonal decision boundaries yield slightly lower AUC (0.8867) than continuous kernel methods. |
| **KNN** | Leverages standardized feature space effectively to form local topological clusters, resulting in improved accuracy (0.8680) and recall (0.9561). |
| **Naive Bayes** | Operates with high training efficiency and solid discrimination (AUC: 0.9210), though conditional feature independence assumptions slightly limit precision (0.8395). |
| **Support Vector Machine** | Outperforms all models by a substantial margin across all 6 metrics (Accuracy: 0.9635, AUC: 0.9949, MCC: 0.9208). The radial basis function (RBF) kernel accurately maps the complex, non-linear physical stability boundaries of the electrical grid. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** — Clearly superior in discriminative power with 0.9949 AUC and 0.9208 MCC. |
