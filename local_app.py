from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'house_price_prediction_secret_key'

# Global variables to store model and preprocessors
model = None
label_encoders = {}
scaler = None
feature_importance = None
model_scores = None

def get_abs_path(*path_parts):
    """Get absolute path relative to the project root"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *path_parts)

def load_model_and_preprocessors():
    """Load the trained model and create preprocessors from the dataset"""
    global model, label_encoders, scaler, feature_importance, model_scores

    try:
        # Load the trained model
        model_path = get_abs_path('models', 'best_model.pkl')
        model = joblib.load(model_path)
        print("Model loaded successfully")

        # Load feature importance
        try:
            feature_importance_path = get_abs_path('models', 'feature_importance.pkl')
            with open(feature_importance_path, 'rb') as f:
                feature_importance = pickle.load(f)
            print("Feature importance loaded successfully")
        except:
            feature_importance = None
            print("Feature importance not available")

        # Load model scores
        try:
            model_scores_path = get_abs_path('models', 'model_scores.pkl')
            with open(model_scores_path, 'rb') as f:
                model_scores = pickle.load(f)
            print("Model scores loaded successfully")
        except:
            model_scores = None
            print("Model scores not available")

        # Load dataset to create preprocessors
        data_path = get_abs_path('data', 'bangalore_house_prices.csv')
        df = pd.read_csv(data_path)

        # Create label encoders for categorical features
        categorical_features = ['area_type', 'availability', 'location', 'size', 'society']

        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(df[feature])
            label_encoders[feature] = le

        # Create scaler for numerical features
        numerical_features = ['total_sqft', 'bath', 'balcony']
        scaler = StandardScaler()
        scaler.fit(df[numerical_features])

        print("Preprocessors created successfully")
        return True

    except Exception as e:
        print(f"Error loading model and preprocessors: {e}")
        return False

def get_unique_values():
    """Get unique values for dropdown options"""
    try:
        data_path = get_abs_path('data', 'bangalore_house_prices.csv')
        df = pd.read_csv(data_path)

        unique_values = {
            'area_type': sorted(df['area_type'].unique()),
            'availability': sorted(df['availability'].unique()),
            'location': sorted(df['location'].unique()),
            'size': sorted(df['size'].unique()),
            'society': sorted(df['society'].unique())
        }
        return unique_values
    except Exception as e:
        print(f"Error getting unique values: {e}")
        return {}

def preprocess_input(input_data):
    """Preprocess input data for prediction"""
    try:
        # Create a copy of input data
        processed_data = input_data.copy()

        # Encode categorical features
        categorical_features = ['area_type', 'availability', 'location', 'size', 'society']
        for feature in categorical_features:
            if feature in processed_data and feature in label_encoders:
                try:
                    processed_data[feature] = label_encoders[feature].transform([processed_data[feature]])[0]
                except ValueError:
                    # Handle unseen categories by using the most frequent category
                    processed_data[feature] = 0

        # Convert to DataFrame for scaling
        df_input = pd.DataFrame([processed_data])

        # Scale numerical features
        numerical_features = ['total_sqft', 'bath', 'balcony']
        df_input[numerical_features] = scaler.transform(df_input[numerical_features])

        # Ensure correct feature order
        feature_order = ['area_type', 'availability', 'location', 'size', 'society', 'total_sqft', 'bath', 'balcony']
        df_input = df_input[feature_order]

        return df_input.values

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def index():
    """Main prediction page"""
    unique_values = get_unique_values()
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        input_data = {
            'area_type': request.form['area_type'],
            'availability': request.form['availability'],
            'location': request.form['location'],
            'size': request.form['size'],
            'society': request.form['society'],
            'total_sqft': float(request.form['total_sqft']),
            'bath': int(request.form['bath']),
            'balcony': int(request.form['balcony'])
        }

        # Preprocess input
        processed_input = preprocess_input(input_data)

        if processed_input is None:
            flash('Error in processing input data', 'error')
            return redirect(url_for('index'))

        # Make prediction
        prediction = model.predict(processed_input)[0]

        # Round to 2 decimal places
        prediction = round(prediction, 2)

        flash(f'Predicted Price: ₹{prediction} Lakhs', 'success')

        # Return to index with prediction
        unique_values = get_unique_values()
        return render_template('index.html',
                               unique_values=unique_values,
                               prediction=prediction,
                               input_data=input_data)

    except Exception as e:
        flash(f'Error in prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Data visualization dashboard"""
    # Get list of visualization files
    viz_dir = get_abs_path('static', 'visualizations')
    visualizations = []

    if os.path.exists(viz_dir):
        for file in os.listdir(viz_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                visualizations.append(file)

    return render_template('dashboard.html', visualizations=visualizations)

@app.route('/performance')
def performance():
    """Model performance page"""
    # Load model performance data
    performance_data = {
        'best_model': 'Gradient Boosting',
        'r2_score': 0.6635,
        'rmse': 56.57,
        'mae': 39.17
    }

    # Add model scores if available
    if model_scores:
        performance_data.update(model_scores)

    return render_template('performance.html',
                           performance_data=performance_data,
                           feature_importance=feature_importance)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()

        # Preprocess input
        processed_input = preprocess_input(data)

        if processed_input is None:
            return jsonify({'error': 'Error in processing input data'}), 400

        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction = round(prediction, 2)

        return jsonify({
            'prediction': prediction,
            'unit': 'Lakhs',
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model and preprocessors on startup
    if load_model_and_preprocessors():
        print("Application initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize application")