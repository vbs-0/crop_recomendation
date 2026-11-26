# crop_recomendation

Five classifiers trained to recommend a crop from seven soil and climate values, with the dataset, the notebook that produced them and the saved models all in the repository.

| Model | Test split (notebook) | Full dataset |
|---|---|---|
| Random Forest | 99.1% | **99.6%** |
| SVM | 98.0% | **99.2%** |
| KNN | 97.5% | 98.6% |
| Logistic Regression | 95.2% | 96.8% |
| Decision Tree | 90.0% | 88.5% |

## Data

`modified_dataset.csv` — the Kaggle crop-recommendation data plus an extra column:

```
N, P, K, temperature, humidity, ph, rainfall, label, deficiency
```

- **features**: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
- **label**: 22 crops (`rice`, `maize`, `chickpea`, … `mothbeans`)
- **deficiency**: a plant-stress label added to the original data

## Use a model

```sh
pip install pandas numpy scikit-learn joblib

python predict.py --model "Random Forest" 90 42 43 20.9 82 6.5 202.9   # -> rice
python predict.py --all                                                # score every model
```

### The one gotcha

**Only SVM was trained on MinMax-scaled features.** The other four were trained on raw values. If you scale everything (or nothing), four of the five models collapse to chance level (~4.5%). `predict.py` keeps the mapping in one place:

```python
NEEDS_SCALING = {"SVM"}
```

From Python:

```python
import joblib, numpy as np
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
model = joblib.load("Random Forest.pkl")

row = np.array([[90, 42, 43, 20.9, 82, 6.5, 202.9]], dtype=float)   # raw, not scaled
print(encoder.inverse_transform(model.predict(row))[0])              # rice
```

## Files

| File | What |
|---|---|
| `final_test.ipynb` | the notebook: heatmap, label encoding, 80/20 split, training, metrics, ROC curve, interactive prediction |
| `modified_dataset.csv` | the data |
| `Decision Tree.pkl` | `DecisionTreeClassifier(criterion="entropy", max_depth=5)` |
| `SVM.pkl` | `SVC` (trained on scaled features) |
| `Logistic Regression.pkl` | `LogisticRegression` |
| `Random Forest.pkl` | `RandomForestClassifier` |
| `KNN.pkl` | `KNeighborsClassifier` |
| `scaler.pkl` | `MinMaxScaler` fitted on the training split |
| `label_encoder.pkl` | `LabelEncoder` for the 22 crops |
| `predict.py` | loads any model with the right preprocessing |

All seven pickle files load with current scikit-learn (checked with 1.9.1).

## Notes on this repository

- The README used to be a 20-byte stub; the accuracy table, the preprocessing rule and `predict.py` were added after loading the models and scoring them against `modified_dataset.csv`.
- Full-dataset numbers are higher than the notebook's test-split numbers because the models have already seen most of those rows. Use the test-split column for an honest comparison.
- `predict.py --all` reproduces the full-dataset column exactly.

## Known limits

- No train/test split script outside the notebook, no saved metrics file, no tests.
- The `deficiency` column is not used by any of the five models.
- Models were trained on a single dataset; there is no validation on data from another source.
