# AI Heart Disease Risk Assessment

AI Heart Disease Risk Assessment is a web app that helps estimate heart disease risk from common clinical measurements. It is designed as a decision-support assistant for learning, screening, and demonstration purposes.

This document explains the project in plain language first, then gives technical details for developers.

## What This Project Does (Non-Technical)

The app asks for 13 health-related inputs (like age, blood pressure, cholesterol, and ECG-related values), then gives:

- A risk score (percentage)
- A risk category (high risk or low risk)
- A visual gauge chart
- A short list of possible risk factors
- A downloadable text report

The app also keeps a temporary session history while the app is open, so you can compare multiple assessments.

## Who This Is For

- Students learning AI in healthcare
- Health researchers doing prototype experiments
- Educators demonstrating machine learning applications
- Developers building healthcare-facing AI interfaces

## Important Medical Note

This is not a diagnosis tool. It should not replace a doctor, clinical tests, or professional medical judgment.

## How The System Works

1. Data is loaded from the UCI Heart Disease dataset (dataset id `45`).
2. Data is cleaned by replacing unknown values and removing incomplete rows.
3. Two machine learning models are trained.
4. The model with better test accuracy is saved as `models/heart_model.pkl`.
5. The Streamlit dashboard loads that saved model.
6. A user enters patient values and gets a prediction with charts and a report.

## Main Features

- Risk prediction from 13 clinical parameters
- Risk gauge visualization (0 to 100%)
- Risk factor highlights (for high cholesterol, blood pressure, etc.)
- Session-level assessment history
- Analytics tab with trend and distribution charts
- Downloadable plain-text report for each assessment

## Project Structure

```text
health-AI-assist/
|-- models/
|   `-- heart_model.pkl
|-- notebook/
|   `-- work.ipynb
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   |-- predict.py
|   |-- feature_engineering.py
|   `-- dashboard/
|       `-- hospital_dashboard.py
|-- requirements.txt
|-- runtime.txt
`-- README.md
```

## File-By-File Explanation

- `src/config.py`: Central paths and dataset URL.
- `src/data_loader.py`: Downloads and combines dataset features and target.
- `src/preprocessing.py`: Cleans data and splits it into train/test sets.
- `src/train_model.py`: Trains Logistic Regression and Random Forest, keeps the better one.
- `src/evaluate_model.py`: Displays feature importance when the chosen model supports it.
- `src/predict.py`: Helper for single-patient prediction from Python code.
- `src/dashboard/hospital_dashboard.py`: Full Streamlit user interface.
- `models/heart_model.pkl`: Saved trained model used by the app.
- `notebook/work.ipynb`: Exploratory analysis notebook.
- `src/feature_engineering.py`: Placeholder for future feature engineering logic.

## Inputs Used By The Model

The dashboard collects these 13 inputs:

- `age`: age in years
- `sex`: female or male
- `cp`: chest pain type
- `trestbps`: resting blood pressure
- `chol`: cholesterol
- `fbs`: fasting blood sugar status
- `restecg`: ECG result category
- `thalach`: max heart rate
- `exang`: exercise-induced angina
- `oldpeak`: ST depression
- `slope`: ST slope category
- `ca`: number of major vessels affected
- `thal`: thalassemia category

## Quick Start (Windows)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ucimlrepo
python .\src\train_model.py
streamlit run .\src\dashboard\hospital_dashboard.py
```

The app opens at `http://localhost:8501`.

## Quick Start (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ucimlrepo
python src/train_model.py
streamlit run src/dashboard/hospital_dashboard.py
```

## Why `ucimlrepo` Is Installed Separately

`data_loader.py` uses `from ucimlrepo import fetch_ucirepo`. If this package is missing, training will fail. Install it with:

```bash
pip install ucimlrepo
```

## How To Use The Dashboard

1. Open the `Risk Assessment` tab.
2. Enter all patient values.
3. Click `Analyze Heart Disease Risk`.
4. Review risk score, recommendation, and factor list.
5. Download report if needed.
6. Open `Analytics Dashboard` to see trends across assessments.
7. Open `Patient History` to review previous entries in this session.

## Dashboard Tabs Explained

- `Risk Assessment`: Main prediction interface and result display.
- `Analytics Dashboard`: Pie chart, trend line, and indicator charts.
- `Patient History`: Expandable record of completed assessments.
- `Information`: Model and dataset summary.

## Technical Notes

- Language: Python
- UI framework: Streamlit
- ML libraries: scikit-learn, numpy, pandas
- Visualization: Plotly and Seaborn/Matplotlib
- Model persistence: joblib
- Data source: UCI Heart Disease dataset

`train_model.py` currently compares:

- Logistic Regression (`max_iter=1000`)
- Random Forest (`random_state=42`)

and saves the best model using test accuracy.

## Current Limitations

- No user authentication
- No permanent database for history
- History is cleared when app restarts or session is reset
- No automated test suite yet
- `feature_engineering.py` is currently empty

## Troubleshooting

### Streamlit says file does not exist

Run from project root using the full relative path:

```powershell
streamlit run src/dashboard/hospital_dashboard.py
```

### Model file not found

Train the model first:

```powershell
python src/train_model.py
```

### `ModuleNotFoundError: ucimlrepo`

Install the package:

```powershell
pip install ucimlrepo
```

## Suggested Future Improvements

- Add unit and integration tests
- Persist patient history in a database
- Add model performance report artifacts
- Improve calibration and class-specific clinical interpretation
- Add role-based access and audit logging

## Disclaimer

This project is for educational and support purposes only. Predictions are probabilistic and can be wrong. Final clinical decisions must be made by qualified healthcare professionals.

## License

Educational and research use.
