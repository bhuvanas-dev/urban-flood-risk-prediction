# Spatiotemporal Machine Learning for Early Prediction of Urban Flood Risk Using Rainfall, Terrain and Drainage Data

A Python and Flask web application that uses an XGBoost-based multi-label classification pipeline to identify recurring urban flood-related labels from rainfall, terrain, drainage, and descriptive location features.

> **Implementation note:** The research topic uses the term *spatiotemporal*. However, the dataset currently used by this implementation does not contain a continuous timestamp series. Therefore, the implemented model predicts descriptive labels from each supplied record; it does not yet forecast flood conditions for a future date or time.

## Project Overview

The system combines descriptive/categorical information with numerical geographical and rainfall-related measurements. Text fields are transformed using TF-IDF, numerical fields are scaled separately, and five independent XGBoost binary classifiers produce multi-label predictions.

### Target labels

- `monitor`
- `low_lying`
- `extreme_rain_history`
- `sparse_drainage`
- `ponding_hotspot`

These are the descriptive labels used in the dataset. They are not converted into generic Low, Moderate, or High risk categories.

## Dashboard

Add a screenshot of the running application to `assets/dashboard.png` in this repository. Once added, it will appear here:

![Urban Flood Risk Prediction application](assets/dashboard.png)

## Model and Dataset

- Training records: **2,400**
- Testing records: **500**
- Additional records held aside: **63**
- Text feature extraction: **TF-IDF** on descriptive/categorical fields
- Numerical preprocessing: **StandardScaler** on seven numerical features
- Classifier: **Five independent XGBoost binary classifiers**
- Reported Exact Match Accuracy: **95.8%** on the test split

The model's per-label scores are model outputs for the dataset labels. They should not be interpreted as guaranteed real-world flood probabilities.

### Input features

**Descriptive/text fields:** city name, administrative ward, DEM source, land use, soil group, storm-drain type, and rainfall source.

**Numerical fields:** latitude, longitude, elevation, drainage density, storm-drain proximity, historical rainfall intensity, and return period.

The original dataset and split CSV files are not included in this public repository. This keeps the source data separate from the code and model files.

## Technology Stack

- **Python** – application and machine-learning implementation
- **Flask** – web application backend
- **XGBoost** – multi-label classification (five binary models)
- **scikit-learn** – TF-IDF vectorization, numerical scaling, and evaluation metrics
- **Pandas and NumPy** – data handling and numerical operations
- **SciPy** – combining sparse text features and numerical features
- **Joblib** – saving and loading trained preprocessing/model artifacts
- **HTML and CSS** – web interface

## System Architecture

```text
User
  |
  v
Web Browser (HTML form and CSS interface)
  |
  v
Flask Application (app.py)
  |
  v
Input Preparation
  |-------------------------------|
  v                               v
TF-IDF transformation             Numerical feature scaling
(descriptive text fields)         (7 numerical fields)
  |                               |
  |---------------+---------------|
                  v
       Combined feature matrix
                  |
                  v
     Five XGBoost binary classifiers
                  |
                  v
       Predicted descriptive labels
       + per-label model scores
                  |
                  v
        Results displayed in browser
```

## Repository Structure

```text
urban-flood-risk-prediction/
├── model/                  # Saved vectorizer, scaler, XGBoost models and evaluation report
├── static/                 # CSS and static web assets
├── templates/              # Flask HTML templates
├── app.py                  # Flask application and prediction route
├── train_model.py          # Model training and evaluation pipeline
├── requirements.txt        # Python dependencies
├── RUN_WINDOWS.bat         # Windows launch helper
├── README.md               # Project documentation
└── .gitignore               # Excludes local environment and private dataset files
```

## Installation and Execution

### Prerequisites

- Python installed on your computer
- Git (to clone the repository)

### 1. Clone the repository

```bash
git clone https://github.com/bhuvanas-dev/urban-flood-risk-prediction.git
cd urban-flood-risk-prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the web application

The repository includes the trained model artifacts in the `model/` folder, so you can start the application without retraining, provided those files are present.

```bash
python app.py
```

Open the local address shown in the terminal. The Flask application normally runs at:

```text
http://127.0.0.1:5000
```

### 5. Retrain the model (optional)

The training script reads `train_2400.csv` and `test_500.csv` from the project root. These split files are excluded from the public repository. To retrain, obtain the permitted source data and place the correctly prepared split files in the project root, then run:

```bash
python train_model.py
```

The training script saves the TF-IDF vectorizer, numerical scaler, XGBoost models, and evaluation report in the `model/` directory.

## Evaluation

The implementation evaluates each target label using precision, recall, and F1-score. It also calculates Exact Match Accuracy, which measures the proportion of test records for which all five predicted labels match the actual labels.

**Reported test Exact Match Accuracy: 95.8%.** Results may vary if the data, preprocessing, or training configuration changes.

## Limitations and Future Work

- The current dataset does not provide a continuous time series, so future-time forecasting is not implemented.
- The public repository does not include the original dataset or split CSV files.
- The model's output scores are classification scores, not calibrated or guaranteed real-world flood probabilities.
- Future work may incorporate timestamped rainfall observations, updated drainage measurements, and additional spatial-temporal validation to support genuine forecasting.

## Author

**Bhuvana S**  
GitHub: [bhuvanas-dev](https://github.com/bhuvanas-dev)
