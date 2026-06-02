from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder='src/templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')

    data = CustomData(
        age=int(request.form['age']),
        sex=int(request.form['sex']),
        cp=int(request.form['cp']),
        trestbps=int(request.form['trestbps']),
        chol=int(request.form['chol']),
        fbs=int(request.form['fbs']),
        restecg=int(request.form['restecg']),
        thalach=int(request.form['thalach']),
        exang=int(request.form['exang']),
        oldpeak=float(request.form['oldpeak']),
        slope=int(request.form['slope']),
        ca=int(request.form['ca']),
        thal=int(request.form['thal'])
    )

    pipeline = PredictPipeline()
    pred, proba = pipeline.predict(data.get_data_as_dataframe())

    result     = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    confidence = f"{proba*100:.1f}%" if pred == 1 else f"{(1-proba)*100:.1f}%"

    return render_template('home.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)