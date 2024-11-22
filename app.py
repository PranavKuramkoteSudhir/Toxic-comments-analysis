from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    """Landing page route"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Home page with the prediction form"""
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                comment_text=request.form.get('comment_text')
            )
            pred_df = data.get_data_as_dataframe()
            print("Dataframe created:", pred_df)
            
            predict_pipeline = PredictPipeline()
            print("Making prediction")
            results = predict_pipeline.predict(pred_df)
            print("Prediction complete:", results)
            
            return render_template('home.html', results=results)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('home.html', error="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)