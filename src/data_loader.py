"""
Data loader for Madrid housing data.

Primary sources:
  1. Kaggle "mirbektoktogaraev/madrid-real-estate-market" (~21k listings, 2018)
  2. Kaggle "kanchana1990/madrid-idealista-property-listings" (~900 listings, more recent)
Fallback: synthetic data generator for development/demo purposes.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_kaggle_dataset() -> str:
    """Download Madrid housing dataset from Kaggle using kagglehub.

    Returns the path to the downloaded CSV file.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
    """
    import kagglehub

    try:
        path = kagglehub.dataset_download("mirbektoktogaraev/madrid-real-estate-market")
    except Exception:
        _check_kaggle_credentials()
        raise
    # The download returns a directory; find the CSV inside
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")


def download_idealista_dataset() -> str:
    """Download Idealista Madrid listings dataset from Kaggle.

    Returns the path to the downloaded CSV file.
    """
    import kagglehub

    try:
        path = kagglehub.dataset_download("kanchana1990/madrid-idealista-property-listings")
    except Exception:
        _check_kaggle_credentials()
        raise
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")


def _check_kaggle_credentials():
    """Raise if Kaggle credentials are not available."""
    kaggle_username = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_KEY")
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not kaggle_username and not os.path.exists(kaggle_json):
        raise EnvironmentError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or place kaggle.json in ~/.kaggle/"
        )


def load_kaggle_data() -> pd.DataFrame:
    """Load and clean the Kaggle Madrid real-estate dataset."""
    csv_path = download_kaggle_dataset()
    df = pd.read_csv(csv_path)

    # Standardise column names (the dataset uses these columns):
    # buy_price, house_type_id, sq_mt_built, n_rooms, n_bathrooms,
    # has_lift, is_exterior, has_parking, floor, neighborhood_id, subtitle
    rename_map = {
        "buy_price": "price",
        "sq_mt_built": "size_m2",
        "n_rooms": "rooms",
        "n_bathrooms": "bathrooms",
        "has_lift": "has_elevator",
        "is_exterior": "is_exterior",
        "has_parking": "has_parking",
        "floor": "floor",
    }
    df = df.rename(columns=rename_map)

    # Keep relevant columns
    keep_cols = [
        "price", "size_m2", "rooms", "bathrooms",
        "has_elevator", "is_exterior", "has_parking", "floor",
        "house_type_id", "neighborhood_id", "subtitle",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Drop rows without price or size
    df = df.dropna(subset=["price", "size_m2"])

    # Remove extreme outliers (price < 10k or > 5M, size < 15 or > 1000)
    df = df[(df["price"] >= 10_000) & (df["price"] <= 5_000_000)]
    df = df[(df["size_m2"] >= 15) & (df["size_m2"] <= 1000)]

    # Fill missing numeric with median
    for col in ["rooms", "bathrooms", "floor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill boolean columns with 0
    for col in ["has_elevator", "is_exterior", "has_parking"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Extract district from neighborhood_id field
    # Format: "Neighborhood 135: San Cristóbal (1308.89 €/m2) - District 21: Villaverde"
    if "neighborhood_id" in df.columns:
        df["district"] = df["neighborhood_id"].apply(_extract_district)
        df["neighborhood"] = df["neighborhood_id"].apply(_extract_neighborhood)
        df = df.drop(columns=["neighborhood_id"])

    # Drop subtitle if present (redundant)
    if "subtitle" in df.columns:
        df = df.drop(columns=["subtitle"])

    df = df.reset_index(drop=True)
    return df


def _extract_district(neighborhood_id):
    """Extract district name from neighborhood_id string.

    Example input: 'Neighborhood 135: San Cristóbal (1308.89 €/m2) - District 21: Villaverde'
    Returns: 'Villaverde'
    """
    if pd.isna(neighborhood_id):
        return "Unknown"
    s = str(neighborhood_id)
    # Look for "District N: <name>"
    if "District" in s:
        part = s.split("District")[-1]
        # Remove the number prefix "21: "
        if ":" in part:
            name = part.split(":", 1)[1].strip()
            return name
    return "Unknown"


def _extract_neighborhood(neighborhood_id):
    """Extract neighborhood name from neighborhood_id string.

    Example input: 'Neighborhood 135: San Cristóbal (1308.89 €/m2) - District 21: Villaverde'
    Returns: 'San Cristóbal'
    """
    if pd.isna(neighborhood_id):
        return "Unknown"
    s = str(neighborhood_id)
    if "Neighborhood" in s and ":" in s:
        part = s.split(":", 1)[1]  # "San Cristóbal (1308.89 €/m2) - District 21: Villaverde"
        if "(" in part:
            name = part.split("(")[0].strip()
            return name
        if " - " in part:
            name = part.split(" - ")[0].strip()
            return name
    return "Unknown"


# Mapping from Idealista address names to (district, neighborhood).
# Addresses that match a neighborhood name directly.
_NEIGHBORHOOD_TO_DISTRICT = {
    "Acacias": "Arganzuela", "Chopera": "Arganzuela", "Delicias": "Arganzuela",
    "Imperial": "Arganzuela", "Legazpi": "Arganzuela", "Palos de Moguer": "Arganzuela",
    "Alameda de Osuna": "Barajas", "Campo de las Naciones-Corralejos": "Barajas",
    "Casco Histórico de Barajas": "Barajas", "Timón": "Barajas",
    "Abrantes": "Carabanchel", "Buena Vista": "Carabanchel", "Comillas": "Carabanchel",
    "Opañel": "Carabanchel", "Pau de Carabanchel": "Carabanchel",
    "Puerta Bonita": "Carabanchel", "San Isidro": "Carabanchel", "Vista Alegre": "Carabanchel",
    "Chueca-Justicia": "Centro", "Huertas-Cortes": "Centro",
    "Lavapiés-Embajadores": "Centro", "Malasaña-Universidad": "Centro",
    "Palacio": "Centro", "Sol": "Centro",
    "Bernabéu-Hispanoamérica": "Chamartín", "Castilla": "Chamartín",
    "Ciudad Jardín": "Chamartín", "El Viso": "Chamartín",
    "Nueva España": "Chamartín", "Prosperidad": "Chamartín",
    "Almagro": "Chamberí", "Arapiles": "Chamberí", "Gaztambide": "Chamberí",
    "Nuevos Ministerios-Ríos Rosas": "Chamberí", "Trafalgar": "Chamberí",
    "Vallehermoso": "Chamberí",
    "Atalaya": "Ciudad Lineal", "Colina": "Ciudad Lineal",
    "Concepción": "Ciudad Lineal", "Costillares": "Ciudad Lineal",
    "Pueblo Nuevo": "Ciudad Lineal", "Quintana": "Ciudad Lineal",
    "San Juan Bautista": "Ciudad Lineal", "San Pascual": "Ciudad Lineal",
    "Ventas": "Ciudad Lineal",
    "Arroyo del Fresno": "Fuencarral", "El Pardo": "Fuencarral",
    "Fuentelarreina": "Fuencarral", "La Paz": "Fuencarral",
    "Las Tablas": "Fuencarral", "Mirasierra": "Fuencarral",
    "Montecarmelo": "Fuencarral", "Peñagrande": "Fuencarral",
    "Pilar": "Fuencarral", "Tres Olivos - Valverde": "Fuencarral",
    "Apóstol Santiago": "Hortaleza", "Canillas": "Hortaleza",
    "Conde Orgaz-Piovera": "Hortaleza", "Palomas": "Hortaleza",
    "Pinar del Rey": "Hortaleza", "Sanchinarro": "Hortaleza",
    "Valdebebas - Valdefuentes": "Hortaleza",
    "Virgen del Cortijo - Manoteras": "Hortaleza",
    "Aluche": "Latina", "Campamento": "Latina", "Cuatro Vientos": "Latina",
    "Los Cármenes": "Latina", "Lucero": "Latina",
    "Puerta del Ángel": "Latina", "Águilas": "Latina",
    "Aravaca": "Moncloa", "Argüelles": "Moncloa", "Casa de Campo": "Moncloa",
    "Ciudad Universitaria": "Moncloa", "El Plantío": "Moncloa",
    "Valdemarín": "Moncloa", "Valdezarza": "Moncloa",
    "Fontarrón": "Moratalaz", "Horcajo": "Moratalaz", "Marroquina": "Moratalaz",
    "Media Legua": "Moratalaz", "Pavones": "Moratalaz", "Vinateros": "Moratalaz",
    "Entrevías": "Puente de Vallecas", "Numancia": "Puente de Vallecas",
    "Palomeras Bajas": "Puente de Vallecas", "Palomeras sureste": "Puente de Vallecas",
    "Portazgo": "Puente de Vallecas", "San Diego": "Puente de Vallecas",
    "Adelfas": "Retiro", "Estrella": "Retiro", "Ibiza": "Retiro",
    "Jerónimos": "Retiro", "Niño Jesús": "Retiro", "Pacífico": "Retiro",
    "Castellana": "Salamanca", "Fuente del Berro": "Salamanca",
    "Goya": "Salamanca", "Guindalera": "Salamanca",
    "Lista": "Salamanca", "Recoletos": "Salamanca",
    "Bellas Vistas": "Tetuán", "Berruguete": "Tetuán",
    "Cuatro Caminos": "Tetuán", "Cuzco-Castillejos": "Tetuán",
    "Valdeacederas": "Tetuán", "Ventilla-Almenara": "Tetuán",
    "12 de Octubre-Orcasur": "Usera", "Almendrales": "Usera",
    "Moscardó": "Usera", "Orcasitas": "Usera", "Pradolongo": "Usera",
    "San Fermín": "Usera", "Zofío": "Usera",
    "Ambroz": "Vicálvaro", "Casco Histórico de Vicálvaro": "Vicálvaro",
    "El Cañaveral - Los Berrocales": "Vicálvaro",
    "Valdebernardo - Valderribas": "Vicálvaro",
    "Casco Histórico de Vallecas": "Villa de Vallecas",
    "Ensanche de Vallecas - La Gavia": "Villa de Vallecas",
    "Santa Eugenia": "Villa de Vallecas",
    "Butarque": "Villaverde", "Los Rosales": "Villaverde",
    "Los Ángeles": "Villaverde", "San Andrés": "Villaverde",
    "San Cristóbal": "Villaverde",
}

# Addresses that are district names (resolve to district, neighborhood="Unknown")
_DISTRICT_ALIASES = {
    "Barrio de Salamanca": "Salamanca",
    "Arganzuela": "Arganzuela", "Barajas": "Barajas",
    "Carabanchel": "Carabanchel", "Centro": "Centro",
    "Chamartín": "Chamartín", "Chamberí": "Chamberí",
    "Ciudad Lineal": "Ciudad Lineal", "Fuencarral": "Fuencarral",
    "Hortaleza": "Hortaleza", "Latina": "Latina", "Moncloa": "Moncloa",
    "Moratalaz": "Moratalaz", "Puente de Vallecas": "Puente de Vallecas",
    "Retiro": "Retiro", "Salamanca": "Salamanca", "Tetuán": "Tetuán",
    "Usera": "Usera", "Vicálvaro": "Vicálvaro",
    "Villa de Vallecas": "Villa de Vallecas", "Villaverde": "Villaverde",
}


def _resolve_idealista_address(address: str) -> tuple[str, str]:
    """Map an Idealista address to (district, neighborhood).

    Returns ("Unknown", "Unknown") if unmappable.
    """
    name = str(address).replace(", Madrid", "").strip()

    # Direct neighborhood match
    if name in _NEIGHBORHOOD_TO_DISTRICT:
        return (_NEIGHBORHOOD_TO_DISTRICT[name], name)

    # District-level match
    if name in _DISTRICT_ALIASES:
        return (_DISTRICT_ALIASES[name], "Unknown")

    return ("Unknown", "Unknown")


def load_idealista_data() -> pd.DataFrame:
    """Load and clean the Idealista Madrid property listings dataset."""
    csv_path = download_idealista_dataset()
    df = pd.read_csv(csv_path)

    # Map address -> district + neighborhood
    resolved = df["address"].apply(_resolve_idealista_address)
    df["district"] = resolved.apply(lambda x: x[0])
    df["neighborhood"] = resolved.apply(lambda x: x[1])

    # Rename columns to match our schema
    # Note: 'sqft' in this dataset is actually square meters (Idealista uses m²)
    df = df.rename(columns={
        "sqft": "size_m2",
        "baths": "bathrooms",
    })

    # Add missing boolean columns with NaN (will be filled later)
    df["has_elevator"] = np.nan
    df["is_exterior"] = np.nan
    df["has_parking"] = np.nan
    df["floor"] = np.nan
    df["house_type_id"] = df["typology"].map({
        "Pisos": "HouseType 1: Pisos",
        "Independientes": "HouseType 2: Independientes",
    })

    # Keep only columns matching our schema
    keep_cols = [
        "price", "size_m2", "rooms", "bathrooms",
        "has_elevator", "is_exterior", "has_parking", "floor",
        "house_type_id", "district", "neighborhood",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop rows without price or size, and remove outliers
    df = df.dropna(subset=["price", "size_m2"])
    df = df[(df["price"] >= 10_000) & (df["price"] <= 10_000_000)]
    df = df[(df["size_m2"] >= 15) & (df["size_m2"] <= 2000)]

    # Drop rows that couldn't be mapped to a known district
    df = df[df["district"] != "Unknown"]

    df = df.reset_index(drop=True)
    return df


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Madrid housing data for development/demo."""
    rng = np.random.default_rng(seed)

    districts = [
        "Centro", "Arganzuela", "Retiro", "Salamanca", "Chamartín",
        "Tetuán", "Chamberí", "Fuencarral-El Pardo", "Moncloa-Aravaca",
        "Latina", "Carabanchel", "Usera", "Puente de Vallecas",
        "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde",
        "Villa de Vallecas", "Vicálvaro", "San Blas-Canillejas", "Barajas",
    ]

    # Price per m2 varies by district (rough Madrid ranges)
    district_price_per_m2 = {
        "Salamanca": 6500, "Chamberí": 5800, "Centro": 5200,
        "Retiro": 5500, "Chamartín": 5600, "Moncloa-Aravaca": 4800,
        "Arganzuela": 4500, "Tetuán": 4000, "Hortaleza": 3800,
        "Ciudad Lineal": 3500, "Fuencarral-El Pardo": 3600,
        "Barajas": 3200, "Moratalaz": 2800, "San Blas-Canillejas": 2700,
        "Latina": 2600, "Carabanchel": 2400, "Usera": 2300,
        "Puente de Vallecas": 2200, "Villa de Vallecas": 2100,
        "Villaverde": 2000, "Vicálvaro": 2100,
    }

    district_choices = rng.choice(districts, size=n_samples)
    sizes = rng.integers(30, 300, size=n_samples).astype(float)
    rooms = np.clip(rng.poisson(lam=2.5, size=n_samples), 1, 8)
    bathrooms = np.clip(rng.poisson(lam=1.2, size=n_samples), 1, 5)
    floors = rng.integers(0, 15, size=n_samples)
    has_elevator = rng.choice([0, 1], size=n_samples, p=[0.2, 0.8])
    is_exterior = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    has_parking = rng.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    # Calculate prices with some noise
    base_prices = np.array([
        district_price_per_m2.get(d, 3000) for d in district_choices
    ]) * sizes

    # Adjustments
    room_factor = 1 + (rooms - 2) * 0.03
    bathroom_factor = 1 + (bathrooms - 1) * 0.05
    elevator_factor = 1 + has_elevator * 0.05
    exterior_factor = 1 + is_exterior * 0.04
    parking_factor = 1 + has_parking * 0.08
    floor_factor = 1 + np.clip(floors, 0, 10) * 0.005

    prices = (
        base_prices * room_factor * bathroom_factor
        * elevator_factor * exterior_factor * parking_factor * floor_factor
    )
    # Add noise (±15%)
    noise = rng.normal(1.0, 0.15, size=n_samples)
    prices = np.clip(prices * noise, 15_000, 5_000_000).astype(int)

    df = pd.DataFrame({
        "price": prices,
        "size_m2": sizes,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "floor": floors,
        "has_elevator": has_elevator,
        "is_exterior": is_exterior,
        "has_parking": has_parking,
        "district": district_choices,
    })

    return df


def load_data(force_synthetic: bool = False) -> pd.DataFrame:
    """Load housing data: try Kaggle first, fall back to synthetic.

    Set force_synthetic=True to skip Kaggle download.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cached = os.path.join(DATA_DIR, "madrid_housing.csv")

    # Return cached data if available
    if os.path.exists(cached) and not force_synthetic:
        return pd.read_csv(cached)

    if not force_synthetic:
        try:
            df = load_kaggle_data()
            print(f"✓ Loaded {len(df)} records from main Kaggle dataset")

            # Try to merge Idealista data
            try:
                df_ideal = load_idealista_data()
                # Fill missing boolean/floor columns with medians from main dataset
                for col in ["has_elevator", "is_exterior", "has_parking", "floor"]:
                    if col in df.columns:
                        median_val = df[col].median()
                        df_ideal[col] = df_ideal[col].fillna(median_val)
                df = pd.concat([df, df_ideal], ignore_index=True)
                print(f"✓ Added {len(df_ideal)} Idealista records → {len(df)} total")
            except Exception as e:
                print(f"⚠ Could not load Idealista data: {e}")

            df.to_csv(cached, index=False)
            return df
        except Exception as e:
            print(f"⚠ Could not load Kaggle data: {e}")
            print("  Falling back to synthetic data.")

    df = generate_synthetic_data()
    df.to_csv(cached, index=False)
    print(f"✓ Generated {len(df)} synthetic records for development")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.describe())
    print(f"\nDistricts: {df['district'].nunique()}")
    print(df["district"].value_counts().head(10))
