# Income Prediction using Machine Learning

## 📌 About the Project

This project predicts whether a person's **annual income is above or below $50,000** based on demographic and employment attributes such as age, education, occupation, marital status, and hours worked per week.

It uses the well-known **UCI Adult Census Income dataset** and applies a complete machine learning workflow — from data cleaning and encoding to exploratory data analysis (EDA) and building a classification model with **Logistic Regression**.

## 🧠 What It Does

- Loads and cleans the raw census income dataset (handles missing values represented as `?`)
- Renames and encodes categorical columns (workclass, education, marital status, occupation, relationship, race, native country, sex) into numeric labels
- Converts the target `income` column into binary classes: `0` for `<=50K` and `1` for `>50K`
- Performs Exploratory Data Analysis (EDA) with visualizations:
  - Income distribution by gender
  - Native country distribution
  - Correlation heatmap between features
  - Boxplots to identify outliers (e.g., in `final_weight`, `capital_gain`)
- Scales the features using `StandardScaler`
- Splits the data into training and testing sets (70/30 split)
- Trains a **Logistic Regression** model to classify income level
- Evaluates the model using:
  - Classification Report (precision, recall, F1-score)
  - Accuracy Score
  - Confusion Matrix (visualized as a heatmap)

## 📊 Dataset

- **Name:** Adult Census Income Dataset (`income_evaluation.csv`)
- **Source:** UCI Machine Learning Repository
- **Target variable:** `income` (`<=50K` or `>50K`)
- **Features:** age, workclass, final_weight, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy` — data handling
  - `matplotlib`, `seaborn` — data visualization
  - `scikit-learn` — preprocessing, model building, and evaluation

## 📈 Model Performance

The Logistic Regression model achieved:

- **Accuracy:** ~78.5%
- Detailed precision, recall, and F1-scores are available in the notebook's classification report, along with a confusion matrix heatmap for a visual breakdown of correct vs incorrect predictions.

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
   ```
2. Install the required libraries
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place the `income_evaluation.csv` dataset in the project folder
4. Open and run the notebook
   ```bash
   jupyter notebook Income_prediction_using_ML.ipynb
   ```

## 📂 Project Structure

```
├── Income_prediction_using_ML.ipynb   # Main notebook with full workflow
├── income_evaluation.csv              # Dataset (Adult Census Income)
└── README.md                          # Project documentation
```

## 🔮 Future Improvements

- Try other classification models (Random Forest, XGBoost, SVM) and compare performance
- Handle class imbalance in the target variable for better recall on the `>50K` class
- Use One-Hot Encoding or target encoding instead of simple label mapping for categorical features
- Perform hyperparameter tuning using GridSearchCV

## 👩‍💻 Authors

**Radhika Pal**
**Anurag Chaudhary**
