# HEART-DISEASE-PREDICTION
# Heart Disease Prediction using Logistic Regression

A machine learning project that predicts the likelihood of heart disease in patients using logistic regression based on various clinical and demographic features.

## 🩺 Dataset Overview

This project uses a comprehensive heart disease dataset with **14 clinical features** to predict cardiovascular disease risk.

### Features Description

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **age** | Numeric | Age of the patient | Years |
| **sex** | Binary | Gender (1 = Male, 0 = Female) | 0, 1 |
| **cp** | Categorical | Chest pain type | 0-3 |
| **trestbps** | Numeric | Resting blood pressure | mmHg |
| **chol** | Numeric | Serum cholesterol level | mg/dl |
| **fbs** | Binary | Fasting blood sugar > 120 mg/dl | 0, 1 |
| **restecg** | Categorical | Resting electrocardiographic results | 0-2 |
| **thalach** | Numeric | Maximum heart rate achieved | bpm |
| **exang** | Binary | Exercise induced angina | 0, 1 |
| **oldpeak** | Numeric | ST depression induced by exercise | Numeric |
| **slope** | Categorical | Slope of peak exercise ST segment | 0-2 |
| **ca** | Numeric | Number of major vessels colored by fluoroscopy | 0-3 |
| **thal** | Categorical | Thalassemia type | 0-3 |
| **target** | Binary | **Heart disease presence (Target)** | 0, 1 |

## 🎯 Project Objective

Develop a **logistic regression model** to predict the probability of heart disease based on patient clinical data, enabling early detection and preventive healthcare measures.

## 📊 Feature Correlation Analysis

The correlation heatmap reveals important relationships:

### Strong Positive Correlations with Heart Disease:
- **cp (Chest Pain Type)**: 0.43 - Certain chest pain types strongly indicate heart disease
- **thalach (Max Heart Rate)**: 0.42 - Higher heart rates during exercise
- **slope**: 0.35 - ST segment slope characteristics

### Strong Negative Correlations with Heart Disease:
- **exang (Exercise Angina)**: -0.44 - Exercise-induced angina
- **oldpeak**: -0.44 - ST depression values
- **ca (Major Vessels)**: -0.38 - Number of vessels with significant blockage
- **thal (Thalassemia)**: -0.34 - Thalassemia types

## 🛠️ Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Logistic regression implementation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization and heatmaps
- **Jupyter Notebook** - Development environment

## 📁 Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart_disease.csv
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_analysis.ipynb
│   └── model_development.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── prediction.py
├── models/
│   └── logistic_regression_model.pkl
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   └── roc_curve.png
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```


## 📈 Model Performance Metrics

### Logistic Regression Evaluation:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for heart disease cases
- **Recall (Sensitivity)**: Ability to identify actual heart disease cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed prediction breakdown

## 🔮 Usage Example

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_csv('data/heart_disease.csv')

# Separate features and target
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]
```

## 🏥 Clinical Applications

### Healthcare Benefits:
- **Early Detection**: Identify high-risk patients before symptoms appear
- **Risk Stratification**: Categorize patients by heart disease probability
- **Treatment Planning**: Guide preventive care and intervention strategies
- **Resource Allocation**: Prioritize screening and monitoring resources
- **Cost Reduction**: Prevent expensive emergency treatments through early intervention

### Target Users:
- **Cardiologists** - Risk assessment and diagnosis support
- **General Practitioners** - Screening and referral decisions
- **Healthcare Systems** - Population health management
- **Researchers** - Clinical studies and epidemiological research

## 📋 Key Insights from Correlation Analysis

1. **Chest Pain Type (cp)** is the strongest predictor of heart disease
2. **Exercise-induced symptoms** (exang, oldpeak) are crucial negative indicators
3. **Age and gender** show moderate correlations with heart disease risk
4. **Cholesterol levels** have weaker correlation than expected
5. **Cardiac stress test results** (thalach, slope) are important predictors

## ⚠️ Model Limitations

- **Medical Disclaimer**: This model is for educational/research purposes only
- **Not for Clinical Diagnosis**: Should not replace professional medical evaluation
- **Data Quality**: Performance depends on accurate and complete input data
- **Population Bias**: Model trained on specific demographic may not generalize

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/medical-improvement`)
3. Commit your changes (`git commit -m 'Add medical feature analysis'`)
4. Push to the branch (`git push origin feature/medical-improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

Joy Dorcas

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- Medical professionals for clinical insight validation
- Scikit-learn community for logistic regression implementation
- Healthcare data science best practices

## 📚 References

- American Heart Association Guidelines
- Clinical cardiology research papers
- Machine learning in healthcare literature
- Logistic regression in medical diagnosis studies

---

**⚠️ IMPORTANT MEDICAL DISCLAIMER**: This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.
