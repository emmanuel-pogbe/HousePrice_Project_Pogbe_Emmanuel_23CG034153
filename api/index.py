"""
House Price Prediction Web Application
Flask-based web interface for the trained Random Forest model
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for loaded model components
model = None
scaler = None
label_encoders = None
metadata = None
selected_features = None

# ============================================================================
# LOAD MODEL COMPONENTS ON STARTUP
# ============================================================================

def load_model_components():
    """Load all saved model components"""
    global model, scaler, label_encoders, metadata, selected_features
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(os.path.join(script_dir, "..", "model"))
        
        # Try loading from model subdirectory first, then fallback to script directory
        for base_path in [model_dir, script_dir]:
            try:
                model = joblib.load(os.path.join(base_path, 'house_price_model.pkl'))
                scaler = joblib.load(os.path.join(base_path, 'feature_scaler.pkl'))
                label_encoders = joblib.load(os.path.join(base_path, 'label_encoders.pkl'))
                metadata = joblib.load(os.path.join(base_path, 'model_metadata.pkl'))
                break
            except FileNotFoundError:
                continue
        
        # If still not found, raise error
        if model is None:
            raise FileNotFoundError("Model files not found")
        
        selected_features = metadata['selected_features']
        
        print("✓ Model components loaded successfully!")
        print(f"  Model Type: {metadata['model_type']}")
        print(f"  Features: {selected_features}")
        print(f"  Test R² Score: {metadata['test_r2_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model components: {str(e)}")
        return False

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive feature data, make prediction, and return result
    Expected JSON format:
    {
        'OverallQual': int,
        'GrLivArea': int,
        'TotalBsmtSF': int,
        'GarageCars': int,
        'YearBuilt': int,
        'Neighborhood': str
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that all required features are present
        if not all(feature in data for feature in selected_features):
            return jsonify({
                'success': False,
                'error': 'Missing required features'
            }), 400
        
        # Create DataFrame with user input
        user_data = pd.DataFrame({
            feature: [data[feature]] for feature in selected_features
        })
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in user_data.columns:
                try:
                    user_data[col] = label_encoders[col].transform(user_data[col])
                except ValueError as e:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid value for {col}: {str(e)}'
                    }), 400
        
        # Scale features
        user_data_scaled = scaler.transform(user_data)
        
        # Make prediction
        predicted_price = model.predict(user_data_scaled)[0]
        
        # Return result
        return jsonify({
            'success': True,
            'predicted_price': float(predicted_price),
            'formatted_price': f"${predicted_price:,.2f}",
            'model_accuracy': f"{metadata['test_r2_score']:.2%}",
            'model_mae': f"${metadata['test_mae']:,.2f}"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/neighborhoods', methods=['GET'])
def get_neighborhoods():
    """Return list of available neighborhoods"""
    try:
        if 'Neighborhood' in label_encoders:
            neighborhoods = sorted(label_encoders['Neighborhood'].classes_.tolist())
            return jsonify({
                'success': True,
                'neighborhoods': neighborhoods
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Neighborhood encoder not found'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Return model information"""
    try:
        return jsonify({
            'success': True,
            'model_type': metadata['model_type'],
            'features': selected_features,
            'test_r2_score': metadata['test_r2_score'],
            'test_rmse': metadata['test_rmse'],
            'test_mae': metadata['test_mae']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("HOUSE PRICE PREDICTION - WEB APPLICATION")
    print("="*80)
    
    # Load model components before starting the app
    if load_model_components():
        print("\n✓ Starting Flask application...")
        print("\nAccess the application at: http://localhost:5000")
        print("="*80 + "\n")
        
        # Run the Flask app
        port = int(os.environ.get("PORT",5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print("\n✗ Failed to load model components. Please check the files.")
        print("  Required files:")
        print("    - house_price_model.pkl")
        print("    - feature_scaler.pkl")
        print("    - label_encoders.pkl")
        print("    - model_metadata.pkl")
