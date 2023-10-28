# Import necessary libraries and modules
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import custom modules for data processing and prediction
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a Flask application instance
application = Flask(__name__)
app = application

# Define a route for the homepage
@app.route('/')
def index():
    # Render the index.html template for the homepage
    return render_template('index.html')

# Define a route for predicting data points
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If the request method is GET, render the home.html template
        return render_template('home.html')
    else:
        # If the request method is POST, extract data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score'),
        )

        # Convert the data to a pandas DataFrame
        pred_df = data.get_data_as_data_frame()

        # Create an instance of the PredictPipeline
        predict_pipeline = PredictPipeline()

        # Use the pipeline to make predictions on the data
        results = predict_pipeline.predict(pred_df)

        # Render the home.html template with the prediction results
        return render_template('home.html', results=results[0])

# Start the Flask application if this script is the main entry point
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)