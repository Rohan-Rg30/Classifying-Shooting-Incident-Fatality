# 🔫 NYPD Shooting Incident — Fatal vs Non-Fatal Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohan-Rg30/Classifying-Shooting-Incident-Fatality/blob/main/Notebook/NYPD_Shooting_Fatality_Prediction.ipynb)

An advanced, end-to-end machine learning binary classification pipeline built entirely in Google Colab to analyze historical New York City shooting incidents and accurately predict whether an incident resulted in a fatality (`STATISTICAL_MURDER_FLAG`).

---

## 📂 Repository Structure
* `notebooks\`: The self-contained master notebook containing the entire workflow, including Exploratory Data Analysis (EDA), categorical encoding, class imbalance remediation, multi-model benchmarking, and the interactive simulator CLI.
* `README.md`: Professional project documentation highlighting technical implementation, methodology, and performance execution.
* `.gitignore`: Configured parameters to maintain a professional workspace clean of hidden local system cache files.

---

## 🛠️ Data Science Pipeline Highlights
* **Strategic Feature Engineering:** Deconstructs raw temporal data (`OCCUR_DATE`, `OCCUR_TIME`) into rich features like Month, Hour, and Day of Week to capture cyclic crime fluctuations.
* **Demographic Interaction Mapping:** Encodes multi-categorical demographic features across Perpetrator and Victim profiles (`AGE_GROUP`, `SEX`, `RACE`) alongside precise geospatial borough data (`BORO`, `PRECINCT`).
* **Robust Imbalance Remediation:** Integrates Synthetic Minority Over-sampling Technique (`SMOTE`) to balance the representation of fatal vs. non-fatal incidents safely within the training partition.
* **Production Validation Framework:** Models are executed via automated pipelines ensuring that imputation, encoding, and standard scaling operations run cleanly without introducing data leakage.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Handling & Processing:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn, Imbalanced-Learn (`SMOTE`)
* **Gradient Boosting Ecosystem:** XGBoost (`xgboost`)
* **Visualizations:** Matplotlib, Seaborn

---

## 📊 Model Performance Evaluation

The models were evaluated using an 80/20 train-test split configuration across three distinct architectures:
1. **XGBoost Classifier** — High-performance tree boosting optimized for capturing complex non-linear combinations of features.
2. **Random Forest** — A powerful bagging ensemble utilized to control model variance and counter overfitting.
3. **Logistic Regression** — A regularized linear baseline model utilized to evaluate linear separability.

### Performance Summary
* **Top Performer:** <span style="color: #00FFFF;">Random Forest</span>
* **Accuracy** | **85.05%**
* **Precision** | **Non-Fatal: 0.84%** | **Fatal: 0.86**
* **Recall** | **Non-Fatal: 0.87%** | **Fatal: 0.83**
* **AUC** | **90.69%**

---

## ⚙️ How to Run the Code Instantly
You do not need to install anything locally on your machine to test this project!

• Click the **Open In Colab** badge at the top of this page.
