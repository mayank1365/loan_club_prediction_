# Lending Club Loan Status Prediction - Neural Network Classification

## 📊 Project Overview

This project implements a deep learning solution for predicting loan status using the Lending Club dataset. The neural network model classifies loans as either "Fully Paid" or "Charged Off" based on borrower characteristics and loan features.

## 🎯 Business Problem

Financial institutions need to assess the risk of loan defaults to make informed lending decisions. This project addresses the critical need to:
- Predict loan outcomes with high accuracy
- Identify key risk factors that influence loan performance
- Minimize financial losses from defaulted loans
- Optimize lending strategies

## 📈 Key Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 80.66% | Good overall performance |
| **Precision** | 81.17% | 81% of predicted good loans are actually good |
| **Recall** | 98.87% | Model catches 99% of all good loans |
| **F1-Score** | 89.15% | Strong balanced performance |
| **AUC-ROC** | 72.50% | Fair discrimination ability |

## 🔧 Technical Stack

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning utilities
- **Matplotlib/Seaborn** - Data visualization

## 📁 Project Structure

```
lending-club-prediction/
├── README.md
├── lending_club_analysis.py
├── lending_club_model.h5
├── feature_importance.csv
├── model_predictions.csv
├── data/
│   └── lending_club_loan_two.csv
├── visualizations/
│   ├── eda_plots.png
│   ├── model_performance.png
│   └── feature_importance.png
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Usage
```python
# Load and run the complete analysis
python lending_club_analysis.py
```

## 📊 Dataset Overview

- **Total Samples**: 396,030 loans
- **Features**: 27 original features, engineered to 101 features
- **Target Variable**: Binary classification (Fully Paid vs Charged Off)
- **Class Distribution**: 
  - Fully Paid: 318,357 (80.4%)
  - Charged Off: 77,673 (19.6%)

### Key Features
- **Financial**: Loan amount, interest rate, annual income, DTI ratio
- **Credit History**: Credit line history, revolving balance, total accounts
- **Borrower Profile**: Employment length, home ownership, verification status
- **Loan Details**: Term, grade, purpose, issue date

## 🔍 Data Preprocessing

### Missing Value Treatment
- **Numerical Features**: Median imputation
- **Categorical Features**: Mode imputation
- **Missing Value Summary**:
  - `emp_title`: 5.79% missing
  - `emp_length`: 4.62% missing
  - `mort_acc`: 9.54% missing
  - `pub_rec_bankruptcies`: 0.14% missing

### Feature Engineering
1. **Date Extraction**: Year from issue date and earliest credit line
2. **Credit History Length**: Calculated from credit timeline
3. **DTI Categorization**: Binned debt-to-income ratios
4. **Categorical Encoding**: Label encoding for low cardinality, one-hot for high cardinality

### Feature Selection
- **101 Total Features** after preprocessing
- **Highly Correlated Pairs**: `loan_amnt` - `installment` (r=0.954)
- **Feature Scaling**: StandardScaler applied to all numerical features

## 🧠 Model Architecture

### Neural Network Design
```python
Sequential([
    Dense(128, activation='relu', input_shape=(101,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)
- **Validation Split**: 20%
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

## 📈 Model Performance

### Confusion Matrix
```
                Predicted
              0      1
Actual   0 |  934  14601
         1 |  718  62953
```

### Performance Breakdown
- **True Positives**: 62,953 (correctly identified good loans)
- **True Negatives**: 934 (correctly identified bad loans)
- **False Positives**: 14,601 (incorrectly predicted as good)
- **False Negatives**: 718 (missed bad loans)

### Business Impact
- **Risk Mitigation**: Model catches 98.9% of good loans, minimizing missed opportunities
- **Precision Trade-off**: 18.8% of predicted good loans may default
- **Overall Accuracy**: 80.7% correct predictions across all loan decisions

## 🔍 Feature Importance Analysis

### Top 10 Most Important Features
1. **Address-related features** (Military addresses) - 1.10-1.15
2. **Issue Year** - 0.88
3. **Annual Income** - 0.77
4. **Grade Encoded** - 0.65
5. **Interest Rate** - 0.52
6. **Term Encoded** - 0.52
7. **DTI Ratio** - 0.49
8. **Revolving Balance** - 0.49
9. **Open Accounts** - 0.42

### Key Insights
- **Temporal factors** (issue year) significantly impact loan performance
- **Borrower income** remains a critical predictor
- **Loan characteristics** (grade, interest rate) strongly influence outcomes
- **Credit utilization** patterns are important risk indicators

## 📊 Exploratory Data Analysis Findings

### Loan Distribution
- **Average Loan Amount**: $14,114
- **Interest Rate Range**: 5.32% - 30.99%
- **Most Common Loan Purpose**: Debt consolidation
- **Loan Grades**: B and C grades most common

### Borrower Characteristics
- **Average Annual Income**: $74,203
- **Employment Length**: 10+ years most common
- **Home Ownership**: Most borrowers rent or have mortgages
- **Credit History**: Average 12.5 years

## ⚠️ Model Limitations

### Class Imbalance Impact
- **High Recall, Lower Precision**: Model optimized to catch good loans
- **False Positive Rate**: 94% for bad loans (concerning for risk management)
- **Recommendation**: Implement cost-sensitive learning or SMOTE

### Feature Limitations
- **Address Bias**: Military addresses dominate importance (potential data leakage)
- **Temporal Bias**: Issue year importance suggests time-dependent patterns
- **Missing Context**: Limited external economic factors

## 🔮 Recommendations for Improvement

### Technical Enhancements
1. **Address Class Imbalance**: Implement SMOTE or cost-sensitive learning
2. **Feature Engineering**: Add macroeconomic indicators
3. **Model Ensemble**: Combine multiple algorithms
4. **Hyperparameter Tuning**: Grid search for optimal parameters
5. **Cross-Validation**: Implement k-fold validation

### Business Applications
1. **Risk-Based Pricing**: Use probability scores for interest rate setting
2. **Loan Approval Threshold**: Adjust based on risk tolerance
3. **Portfolio Management**: Monitor feature importance shifts over time
4. **Regulatory Compliance**: Ensure model fairness across demographics

## 📋 Model Deployment Considerations

### Production Requirements
- **Model Versioning**: Track model performance over time
- **Feature Drift Monitoring**: Monitor input feature distributions
- **Performance Monitoring**: Track prediction accuracy on new data
- **A/B Testing**: Compare model performance against current systems

### Interpretability
- **SHAP Values**: Implement for individual prediction explanations
- **Feature Importance**: Regular updates as business evolves
- **Model Documentation**: Maintain clear model cards for stakeholders

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

