from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from MLProject.pipeline.prediction import PredictionPipeline
from MLProject import logger
from datetime import datetime

app = Flask(__name__)

# Define AQI bucket logic
def get_aqi_bucket(aqi_score):
    if 0 <= aqi_score <= 50:
        return "Good"
    elif 51 <= aqi_score <= 100:
        return "Satisfactory"
    elif 101 <= aqi_score <= 200:
        return "Moderate"
    elif 201 <= aqi_score <= 300:
        return "Poor"
    elif 301 <= aqi_score <= 400:
        return "Very Poor"
    elif 401 <= aqi_score <= 500:
        return "Severe"
    else:
        return "Extreme"

# Define pollutant constraints for server-side validation
POLLUTANT_CONSTRAINTS = {
    'PM2.5': {'min': 0.0, 'max': 1000.0, 'unit': 'µg/m³'},
    'PM10': {'min': 0.0, 'max': 2000.0, 'unit': 'µg/m³'},
    'NO': {'min': 0.0, 'max': 250.0, 'unit': 'µg/m³'},
    'NO2': {'min': 1.0, 'max': 250.0, 'unit': 'µg/m³'},
    'NOx': {'min': 1.0, 'max': 500.0, 'unit': 'µg/m³'},
    'NH3': {'min': 0.0, 'max': 300.0, 'unit': 'µg/m³'},
    'CO': {'min': 0.0, 'max': 20.0, 'unit': 'µg/m³'}, # Changed min from 30.0 to 5.0
    'SO2': {'min': 1.0, 'max': 150.0, 'unit': 'µg/m³'},
    'O3': {'min': 1.0, 'max': 150.0, 'unit': 'µg/m³'},
    'Benzene': {'min': 0.0, 'max': 60.0, 'unit': 'µg/m³'},
    'Toluene': {'min': 1.0, 'max': 120.0, 'unit': 'µg/m³'}, # Widened range from 7.3-369.0
    'Xylene': {'min': 1.0, 'max': 60.0, 'unit': 'µg/m³'}, # Widened range from 3.0-380.0
}

# Define date constraints for server-side validation
MIN_DATE = datetime(2015, 1, 1).date()
MAX_DATE = datetime(2020, 12, 31).date()

@app.route('/', methods=['GET'])
def homePage():
    logger.info("Home page requested.")
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predictRoute():
    try:
        # Prepare data from form
        raw_data = {}
        for key in request.form:
            raw_data[key] = request.form[key]

        # SERVER-SIDE VALIDATION
        validation_errors = []

        # Validate Date
        input_date_str = raw_data.get('Date')
        if not input_date_str:
            validation_errors.append("Date is required.")
        else:
            try:
                input_date = datetime.strptime(input_date_str, '%Y-%m-%d').date()
                if not (MIN_DATE <= input_date <= MAX_DATE):
                    validation_errors.append(f"Date must be between {MIN_DATE.strftime('%Y-%m-%d')} and {MAX_DATE.strftime('%Y-%m-%d')}.")
            except ValueError:
                validation_errors.append("Invalid Date format. Please use YYYY-MM-DD.")

        # Validate Pollutants
        for pollutant, constraints in POLLUTANT_CONSTRAINTS.items():
            value_str = raw_data.get(pollutant)
            if value_str:
                try:
                    value = float(value_str)
                    if not (constraints['min'] <= value <= constraints['max']):
                        validation_errors.append(f"{pollutant} must be between {constraints['min']} and {constraints['max']} {constraints['unit']}.")
                except ValueError:
                    validation_errors.append(f"Invalid value for {pollutant}. Please enter a number.")
            # else: You might also add a check here if a pollutant is mandatory

        if validation_errors:
            logger.warning(f"Validation errors received: {'; '.join(validation_errors)}")
            return render_template('results.html',
                                   prediction="Validation Error",
                                   aqi_bucket="Input Error",
                                   error_message="Please correct the following issues:<br>" + "<br>".join(validation_errors))

        # Construct DataFrame for PredictionPipeline
        data_for_pipeline = {
            'City': [raw_data.get('City')],
            'Date': [raw_data.get('Date')], # Pass string; PredictionPipeline will parse
            'PM2.5': [float(raw_data.get('PM2.5')) if raw_data.get('PM2.5') else np.nan],
            'PM10': [float(raw_data.get('PM10')) if raw_data.get('PM10') else np.nan],
            'NO': [float(raw_data.get('NO')) if raw_data.get('NO') else np.nan],
            'NO2': [float(raw_data.get('NO2')) if raw_data.get('NO2') else np.nan],
            'NOx': [float(raw_data.get('NOx')) if raw_data.get('NOx') else np.nan],
            'NH3': [float(raw_data.get('NH3')) if raw_data.get('NH3') else np.nan],
            'CO': [float(raw_data.get('CO')) if raw_data.get('CO') else np.nan],
            'SO2': [float(raw_data.get('SO2')) if raw_data.get('SO2') else np.nan],
            'O3': [float(raw_data.get('O3')) if raw_data.get('O3') else np.nan],
            'Benzene': [float(raw_data.get('Benzene')) if raw_data.get('Benzene') else np.nan],
            'Toluene': [float(raw_data.get('Toluene')) if raw_data.get('Toluene') else np.nan],
            'Xylene': [float(raw_data.get('Xylene')) if raw_data.get('Xylene') else np.nan]
        }

        input_df = pd.DataFrame(data_for_pipeline)

        logger.info(f"Received prediction request with data: {input_df.to_dict(orient='records')}")

        obj = PredictionPipeline()
        predicted_aqi = obj.predict(input_df)[0]

        # Round the predicted AQI for display
        predicted_aqi_rounded = round(predicted_aqi, 2)

        # Get the AQI bucket
        aqi_bucket = get_aqi_bucket(predicted_aqi_rounded)

        logger.info(f"Prediction successful. Predicted AQI: {predicted_aqi_rounded} (Bucket: {aqi_bucket})")

        return render_template('results.html',
                               prediction=predicted_aqi_rounded,
                               aqi_bucket=aqi_bucket,
                               error_message="")

    except Exception as e:
        logger.exception(f"Error occurred during prediction: {e}")
        return render_template('results.html',
                               prediction="Error during prediction.",
                               aqi_bucket="System Error",
                               error_message=f"An unexpected error occurred: {e}. Please check server logs.")

if __name__ == "__main__":
    # Ensure logs directory exists if running app.py directly
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    app.run(host='0.0.0.0', port=8080)