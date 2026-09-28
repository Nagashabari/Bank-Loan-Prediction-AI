# Bank Loan Prediction AI — Intelligent Loan Risk Assessment

Bank Loan Prediction AI is an upgraded Flask + scikit-learn portfolio project built on the supplied loan-approval model. It turns a basic prediction page into a complete financial decision-support workspace.

## Included
- Registration, login, logout and profile
- Personal dashboard and application history
- Loan assessment for multiple loan categories
- ML approval probability and rejection probability
- Model risk score / risk band
- Repayment capacity, EMI, DTI and affordability calculations
- Recommended loan amount and indicative-rate estimates (clearly labeled as estimates)
- Credit-score-style project estimate (not a bureau score)
- Feature importance visualization from the trained Random Forest
- Positive factors, risk factors and personalized improvement suggestions
- What-if financial lab / EMI calculator
- Interactive Chart.js analytics
- Anomaly screening flag for unusual application patterns (not a fraud verdict)
- Save applications
- Downloadable PDF assessment reports
- Smart Financial Assistant using local rule-based guidance
- Modern responsive UI with local SVG illustrations

## ML boundary
The existing `model.pkl` is the ML component. It predicts loan approval status from the original 13-feature representation. Financial calculations and recommendation modules are separate decision-support logic; they are not presented as bank underwriting models.

## Run
Use the same Python environment that can load the supplied `model.pkl` (the original model was trained with scikit-learn 1.3.0).

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your normal browser.

## First use
1. Create an account.
2. Login.
3. Open **New Assessment**.
4. Enter applicant information.
5. Review the result, charts, risk factors and financial analysis.
6. Save the application or download the PDF report.

## Important
This is an academic/portfolio decision-support application. Model probabilities are not real bank approvals, credit-bureau scores or regulated financial advice.
