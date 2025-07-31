import joblib
import pandas as pd
from text_cleaner import clean_text  # Ensure it's imported correctly

# Load pipeline
pipeline = joblib.load("model/sentiment_pipeline_model.joblib")

# Test samples
samples = [
    "I absolutely loved this product!",
    "This is the worst thing I’ve ever bought",
    "It’s okay, nothing special"
]

# Convert to Series so .apply() works inside TextCleaner
samples_series = pd.Series(samples)

# Predict
preds = pipeline.predict(samples_series)

# Print predictions
print("Predictions:", preds)
