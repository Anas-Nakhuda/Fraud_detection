# Financial Fraud Detection

A machine learning project that detects fraudulent financial transactions using a Random Forest Classifier. The dataset contains over 6.3 million transactions and suffers from extreme class imbalance (only ~0.13% are fraudulent). 

To ensure efficient computation while preserving model integrity, a stratified 3% sample was used for model training and evaluation.

## Dataset Overview
The dataset contains 11 columns tracking transaction details:
- `step`: Maps a unit of time in the real world (1 step = 1 hour).
- `type`: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
- `amount`: Amount of the transaction in local currency.
- `nameOrig` / `nameDest`: Customer identifiers (dropped during preprocessing)[cite: 1].
- `oldbalanceOrg` / `newbalanceOrig`: Initial and new balances for the origin account[cite: 1].
- `oldbalanceDest` / `newbalanceDest`: Initial and new balances for the destination account[cite: 1].
- `isFraud`: Target variable (1 if fraudulent, 0 otherwise)[cite: 1].

## Project Workflow

### 1. Data Preprocessing & Cleaning
- **Feature Selection:** Dropped unique string identifiers `nameOrig` and `nameDest` to prevent model overfitting[cite: 1].
- **Data Cleaning:** Identified and removed anomalous rows (e.g., transactions where transfers occurred but account balances remained completely unchanged)[cite: 1].
- **Categorical Encoding:** Applied `LabelEncoder` to convert the transaction `type` column into numerical values[cite: 1].

### 2. Feature Engineering
Two critical features were engineered to capture balance discrepancies, which significantly improved the model's ability to isolate fraud[cite: 1]:
- `errorBalanceOrig` = `oldbalanceOrg` - `newbalanceOrig` - `amount`[cite: 1]
- `errorBalanceDest` = `newbalanceDest` - `oldbalanceDest` - `amount`[cite: 1]

### 3. Sampling & Validation Strategy
Due to the dataset's massive size (6.3M+ rows), a **3% stratified sample** was extracted to train the model efficiently without losing the original distribution of fraud vs. non-fraud transactions[cite: 1]. 
- **Train/Test Split:** 80% train, 20% test[cite: 1].
- **Stratification:** Maintained the exact fraud ratio in both splits[cite: 1].

### 4. Model & Evaluation
A `RandomForestClassifier` was trained with `class_weight='balanced'` to handle the remaining internal imbalance[cite: 1].

**Performance Metrics:**
- **Accuracy:** 100%[cite: 1]
- **ROC-AUC Score:** 1.00[cite: 1]
- **Classification Report:**[cite: 1]
```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     38126
           1       1.00      1.00      1.00        50

**Author:** Mohd Anas Nakhuda
    accuracy                           1.00     38176
   macro avg       1.00      1.00      1.00     38176
weighted avg       1.00      1.00      1.00     38176
