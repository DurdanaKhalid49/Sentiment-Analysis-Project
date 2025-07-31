from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load("model/sentiment_pipeline_model.joblib")

# Label map
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['user_input']
        input_series = pd.Series([input_text])
        prediction = pipeline.predict(input_series)[0]
        sentiment = label_map[prediction]
        return render_template('index.html', input_text=input_text, prediction=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
