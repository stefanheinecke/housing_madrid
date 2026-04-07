# Madrid Housing Price Predictor 🏠

AI-powered web application to estimate fair market values for apartments and houses in Madrid.

## Features

- **Gradient Boosting regression model** trained on real Madrid property listings
- **Partial input support** — enter any combination of features; missing values are filled automatically
- **Confidence range** showing estimated price bounds
- **Price per m²** calculation
- **21 Madrid districts** with location-aware pricing
- Clean, responsive web interface

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Kaggle credentials

The app uses the [Madrid Real Estate Market](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market) dataset from Kaggle. To download it automatically:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → "Create New Token"
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<you>\.kaggle\` (Windows)

If Kaggle credentials are not configured, the app will generate **synthetic data** for development/demo purposes. The synthetic data uses realistic Madrid district pricing and property distributions.

### 4. Run the application

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

On first start, the app will:
1. Download data from Kaggle (or generate synthetic data)
2. Train the ML model (~10 seconds)
3. Start the web server

## Project Structure

```
housing_madrid/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Kaggle download + synthetic data generator
│   └── model.py            # ML model (training, prediction, persistence)
├── templates/
│   └── index.html          # Web frontend
├── data/                   # Cached CSV data (git-ignored)
└── models/                 # Trained model files (git-ignored)
```

## Property Features

| Feature | Type | Description |
|---------|------|-------------|
| `size_m2` | numeric | Built area in square meters |
| `rooms` | numeric | Number of rooms/bedrooms |
| `bathrooms` | numeric | Number of bathrooms |
| `floor` | numeric | Floor number (0 = ground) |
| `district` | categorical | Madrid district (21 districts) |
| `has_elevator` | boolean | Building has elevator |
| `is_exterior` | boolean | Exterior-facing apartment |
| `has_parking` | boolean | Parking space included |

## API Endpoints

### `POST /api/predict`
Predict price from property features. All fields are optional (at least one required).

```json
{
    "size_m2": 80,
    "rooms": 3,
    "bathrooms": 2,
    "district": "Salamanca",
    "has_elevator": 1
}
```

Response:
```json
{
    "predicted_price": 520000.00,
    "price_lower": 440000.00,
    "price_upper": 600000.00,
    "price_per_m2": 6500.00,
    "features_used": { ... },
    "defaults_applied": ["floor", "is_exterior", "has_parking"]
}
```

### `GET /api/model-info`
Returns model metrics, available districts, and feature defaults.

### `POST /api/retrain`
Re-download data and retrain the model.

## Data Sources

- **Primary:** [Kaggle — Madrid Real Estate Market](https://www.kaggle.com/datasets/mirbektoktogaraev/madrid-real-estate-market) (~21,000 listings with prices, sizes, locations, and amenities)
- **Fallback:** Synthetic data generator with realistic Madrid district pricing

## Model Details

- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Preprocessing:** StandardScaler for numeric features, OneHotEncoder for districts
- **Hyperparameters:** 300 estimators, max depth 5, learning rate 0.1
- **Evaluation:** 80/20 train/test split with R², MAE, and MAPE metrics

## Using Your Own Data

Place a CSV file at `data/madrid_housing.csv` with these columns:

```
price,size_m2,rooms,bathrooms,floor,has_elevator,is_exterior,has_parking,district
350000,75,3,1,4,1,1,0,Centro
...
```

Then restart the app or call `POST /api/retrain`.

## License

MIT
