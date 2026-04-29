# Audio Classification: Bird vs Cat vs Dog

This project classifies short `.wav` audio clips into **bird**, **cat**, or **dog** using extracted acoustic features.

## What is included

- `src/train_audio_models.py` — feature extraction, preprocessing, model training, evaluation, and plots
- `slides/audio_classification_presentation.pptx` — presentation slides
- `results/` — saved model results and generated figures

## Models trained

- KNN
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

## Preprocessing handled

- Missing values: checked and median imputation included in every model pipeline
- Outliers: numeric features clipped at the 1st/99th percentiles
- Categorical column: labels encoded with `LabelEncoder`
- Imbalanced data: classes were almost balanced; used stratified split and class weights where supported

## Best result

Random Forest performed best by accuracy, reaching about **86.9% accuracy** on the holdout test set.

## How to run

Download the Kaggle dataset and keep the folder structure like this:

```text
Animals/
  bird/*.wav
  cat/*.wav
  dog/*.wav
```

Then run:

```bash
pip install -r requirements.txt
python src/train_audio_models.py --data-dir Animals --out-dir results
```
