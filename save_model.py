# save_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from text_cleaner import TextCleaner

# 1. Load Data
df = pd.read_csv("data/Reviews_cleaned.csv")
df.dropna(subset=['Text', 'Sentiment'], inplace=True)

# 2. Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Sentiment'])  # Save this mapping for dashboard

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['label_encoded'], test_size=0.2, stratify=df['label_encoded'], random_state=42)

# 4. Build Pipeline
vectorizer = TfidfVectorizer(max_features=1000)
model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)

pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# 5. Train & Save
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model/sentiment_pipeline_model.joblib")
joblib.dump(le, "model/label_encoder.joblib")

print("✅ Model pipeline and label encoder saved.")
