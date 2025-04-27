# Set Matplotlib backend to Agg before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model dictionary
try:
    model_data = joblib.load('sales_prediction_model (2).pkl')
    model = model_data['model']
except FileNotFoundError:
    print("Error: Model file 'sales_prediction_model (2).pkl' not found. Please check the file path.")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html', status_message="Please fill the details.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch and validate form data
        form_data = {
            'Item_Identifier': request.form['Item_Identifier'],
            'Item_Weight': float(request.form['Item_Weight']),
            'Item_Fat_Content': request.form['Item_Fat_Content'],
            'Item_Visibility': float(request.form['Item_Visibility']),
            'Item_Type': request.form['Item_Type'],
            'Item_MRP': float(request.form['Item_MRP']),
            'Outlet_Identifier': request.form['Outlet_Identifier'],
            'Outlet_Establishment_Year': int(request.form['Outlet_Establishment_Year']),
            'Outlet_Size': request.form['Outlet_Size'],
            'Outlet_Location_Type': request.form['Outlet_Location_Type'],
            'Outlet_Type': request.form['Outlet_Type']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Make prediction for the original input
        prediction = model.predict(input_df)[0]

        # Define all possible outlet sizes
        outlet_sizes = ['Small', 'Medium', 'High']
        actual_size = form_data['Outlet_Size']

        # Create scenarios for the other two outlet sizes
        scenarios = []
        predictions = []
        for size in outlet_sizes:
            scenario_data = form_data.copy()
            scenario_data['Outlet_Size'] = size
            scenario_df = pd.DataFrame([scenario_data])
            pred = model.predict(scenario_df)[0]
            scenarios.append(f'{size} Outlet')
            predictions.append(pred)

        # Create a comparison bar chart
        plt.figure(figsize=(8, 5))
        bars = plt.bar(scenarios, predictions, color=['blue', 'orange', 'green'])
        plt.title('Sales Prediction Across Different Outlet Sizes')
        plt.ylabel('Predicted Sales (₹)')

        # Add value labels on top of bars
        for bar, pred in zip(bars, predictions):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'₹{round(yval, 2)}', ha='center', va='bottom')

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return render_template('index.html', prediction=round(prediction, 2), image_url=image_base64, status_message="Prediction Successful!")

    except Exception as e:
        return render_template('index.html', status_message=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)