URBAN FLOOD RISK PREDICTION - COMPLETE STARTER IMPLEMENTATION

PROJECT PIPELINE
1. Dataset: original Excel dataset supplied by the student.
2. Split: 2,400 training records + 500 testing records; 63 records held aside.
3. Feature extraction: TF-IDF on descriptive text/categorical fields only.
4. Numeric features: seven numeric columns are scaled separately.
5. Classifier: five independent XGBoost binary classifiers form a multi-label predictor.
6. Output: original descriptive labels: monitor, low_lying, extreme_rain_history,
   sparse_drainage, ponding_hotspot. No Low/Moderate/High conversion.

IMPORTANT DATASET NOTE
The dataset contains 177 distinct label strings/tokens, including many unique
event_YYYY-MM-DD identifiers. The five recurring descriptive tags are modeled.
Date-specific event identifiers are retained in the CSV files but are not
treated as predictive classes because they identify individual dates and are
too sparse to generalize. Confirm this treatment with your lecturer.
The supplied table has no continuous timestamp series, so this implementation
does not claim to forecast a future date/time. It predicts dataset labels from
the provided record features.

INSTALLATION (Windows)
1. Extract this ZIP into a folder.
2. Open that folder in VS Code.
3. Open Terminal > New Terminal.
4. Run: py -m venv venv
5. Run: venv\Scripts\activate
6. Run: python -m pip install --upgrade pip
7. Run: pip install -r requirements.txt

TRAIN MODEL
Run: python train_model.py
Wait until it prints per-label Precision, Recall, F1 and Exact-match accuracy.
Trained model files are saved in model/.

START WEBSITE
Run: python app.py
Open the local address shown in terminal (usually http://127.0.0.1:5000).

FILES
train_2400.csv, test_500.csv, unused_63.csv: reproducible split
train_model.py: TF-IDF + numeric preprocessing + XGBoost training/evaluation
app.py: Flask backend
templates/index.html: input form and prediction display
static/style.css: website styling
model/: generated after successful training

ACADEMIC CAUTION
TF-IDF is conventionally a text feature extraction method. Here it is used only
on the categorical/descriptive text fields because the lecturer selected TF-IDF.
Numeric measurements are handled separately. Do not describe numeric values
as TF-IDF features. Model scores are not guaranteed; report actual output from
your own run and do not invent accuracy.
