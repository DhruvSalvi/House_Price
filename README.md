
# House Price Prediction Flask Web Application

A comprehensive Flask web application for predicting house prices in Bangalore using machine learning.

## Features

### 🏠 Main Prediction Page
- Interactive form with all 8 input fields (area_type, availability, location, size, society, total_sqft, bath, balcony)
- Real-time input validation and user-friendly interface
- Instant price predictions using trained Gradient Boosting model
- Responsive design that works on all devices

### 📊 Data Visualization Dashboard
- Comprehensive visualizations including:
  - Price distribution analysis
  - Feature correlation matrix
  - Price analysis by features
  - Location-wise analysis
  - Model performance comparison
  - Feature importance charts
  - Actual vs predicted prices
  - Residuals analysis
- Interactive charts with zoom functionality
- Links to interactive HTML visualizations

### 🏆 Model Performance Page
- Detailed model metrics and performance analysis
- Comparison of all trained models (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Best model: Gradient Boosting with 66.35% R² score
- Technical implementation details
- Key insights and limitations

### 🎨 Modern UI/UX Design
- Beautiful gradient backgrounds and modern styling
- Bootstrap 5 integration for responsive design
- Font Awesome icons throughout the interface
- Smooth animations and hover effects
- Professional color scheme and typography

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Machine Learning**: scikit-learn, Gradient Boosting Regressor
- **Data Processing**: pandas, numpy
- **Visualization**: Static PNG charts and interactive HTML plots

## Model Performance

- **Best Model**: Gradient Boosting Regressor
- **R² Score**: 66.35%
- **RMSE**: ₹56.57 Lakhs
- **MAE**: ₹39.17 Lakhs
- **Dataset**: 5,000 Bangalore property records
- **Features**: 8 input features (5 categorical, 3 numerical)

## API Endpoints

### Web Routes
- `/` - Main prediction page
- `/dashboard` - Data visualization dashboard
- `/performance` - Model performance analysis

### API Routes
- `POST /api/predict` - JSON API for predictions
- `POST /predict` - Form-based prediction endpoint

## Installation & Setup

1. **Prerequisites**
   ```bash
   pip install flask pandas scikit-learn joblib numpy
   ```

2. **Run the Application**
   ```bash
   cd flask_app
   python app.py
   ```

3. **Access the Application**
   - Open browser and navigate to `http://localhost:5000`
   - The application will be available on all network interfaces

## Usage Examples

### Web Interface
1. Navigate to the main page
2. Fill in all property details in the form
3. Click "Predict Price" to get instant results
4. Explore visualizations in the Dashboard
5. View model performance metrics in Performance page

### API Usage
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_type": "Super built-up Area",
    "availability": "Ready To Move",
    "location": "HSR Layout Sector 6",
    "size": "3 BHK",
    "society": "Brigade Orchards",
    "total_sqft": 1452,
    "bath": 2,
    "balcony": 1
  }'
```

Response:
```json
{
  "prediction": 379.19,
  "status": "success",
  "unit": "Lakhs"
}
```

## File Structure

```
flask_app/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Main prediction page
│   ├── dashboard.html    # Visualization dashboard
│   └── performance.html  # Model performance page
├── static/
│   └── visualizations/   # Static chart images and interactive HTML
├── models/
│   ├── best_model.pkl    # Trained Gradient Boosting model
│   ├── feature_importance.pkl
│   └── model_scores.pkl
└── README.md
```

## Key Features

### Data Preprocessing
- Automatic label encoding for categorical features
- Standard scaling for numerical features
- Robust handling of unseen categories
- Feature engineering and validation

### Error Handling
- Comprehensive input validation
- Graceful error handling for edge cases
- User-friendly error messages
- Fallback mechanisms for missing data

### Security
- Input sanitization and validation
- CSRF protection with secret key
- Safe file handling and path management

### Performance
- Efficient model loading and caching
- Optimized preprocessing pipeline
- Fast prediction response times
- Minimal memory footprint

## Model Details

### Input Features
1. **Area Type** (Categorical): Super built-up Area, Built-up Area, Plot Area, Carpet Area
2. **Availability** (Categorical): Ready To Move, 1 Year, 6 Months, Under Construction, 2 Years, etc.
3. **Location** (Categorical): 48 different locations in Bangalore
4. **Size** (Categorical): 1 BHK, 2 BHK, 3 BHK, 4 BHK, 5 BHK
5. **Society** (Categorical): 36 different societies/builders
6. **Total Sqft** (Numerical): Property area in square feet
7. **Bathrooms** (Numerical): Number of bathrooms
8. **Balconies** (Numerical): Number of balconies

### Output
- **Price**: Predicted property price in Lakhs (₹)

## Future Enhancements

- Real-time market data integration
- Additional cities and regions
- Advanced filtering and search capabilities
- User authentication and saved predictions
- Mobile app development
- Integration with real estate APIs

## License

This project is for educational and demonstration purposes.

---

**Built with ❤️ using Flask and Machine Learning**
