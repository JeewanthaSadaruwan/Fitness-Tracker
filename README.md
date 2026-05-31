# Fitness Tracker - Barbell Exercise Classification

A machine learning project for tracking and classifying barbell exercises using accelerometer and gyroscope data from MetaMotion wearable sensors. The system automatically recognizes different exercise types (bench press, deadlift, squat, overhead press, and barbell row) along with their intensity levels.

## 📋 Project Overview

This project uses time-series sensor data to:
- Process and clean accelerometer and gyroscope measurements
- Extract meaningful features from raw sensor data
- Build machine learning models to classify different barbell exercises
- Analyze exercise patterns and intensities

The pipeline includes data preprocessing, outlier detection, feature engineering (temporal and frequency domain), dimensionality reduction, and multi-class classification.

## 🎯 Exercises Tracked

The system can identify the following exercises:
- **Bench Press** - Heavy and medium variations
- **Deadlift** - Heavy and medium variations  
- **Squat** - Heavy and medium variations
- **Overhead Press (OHP)** - Heavy and medium variations
- **Barbell Row** - Heavy and medium variations

## 📊 Dataset

**Source**: MetaMotion wearable sensor data
- **Sensors**: Accelerometer (12.5 Hz) and Gyroscope (25 Hz)
- **Participants**: Multiple participants (labeled A-E)
- **Files**: 187 CSV files containing raw sensor measurements
- **Features**: 6 axes (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)

### Data Structure
```
data/
├── raw/MetaMotion/          # Raw CSV files from sensors
├── interim/                  # Intermediate processed data
└── processed/               # Final feature-engineered datasets
    ├── metamotion_processed.csv
    └── metamotion_features.csv
```

## 🏗️ Project Structure

```
fitness-tracker/
├── data/                    # Data storage
│   ├── raw/                # Original sensor data
│   ├── interim/            # Intermediate processing steps
│   └── processed/          # Final processed datasets
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code modules
│   ├── data/              
│   │   └── make_dataset.py         # Data loading and preprocessing
│   ├── features/
│   │   ├── build_features.py       # Feature engineering pipeline
│   │   ├── DataTransformation.py   # Low-pass filter, PCA
│   │   ├── TemporalAbstraction.py  # Time-domain features
│   │   ├── FrequencyAbstraction.py # Frequency-domain features
│   │   └── remove_outliers.py      # Outlier detection
│   ├── models/
│   │   ├── train_model.py          # Model training
│   │   ├── predict_model.py        # Predictions
│   │   └── LearningAlgorithms.py   # ML algorithms
│   └── visualization/
│       └── visualize.py            # Plotting utilities
├── models/                 # Trained model artifacts
├── reports/               
│   └── figures/           # Generated visualizations
├── train_model.ipynb      # Model training notebook
├── remove_outliers.ipynb  # Outlier analysis notebook
├── environment.yml        # Conda environment
└── requirements.txt       # Python dependencies
```

## 🛠️ Technologies & Libraries

- **Python 3.8.15**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Signal Processing**: SciPy (Butterworth filter)
- **Environment**: Conda

## 🚀 Getting Started

### Prerequisites
- Anaconda or Miniconda installed
- Python 3.8+

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fitness-tracker
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

3. **Alternative: Use pip**
```bash
pip install -r requirements.txt
```

### Environment Management

```bash
# Update environment
conda env update --file environment.yml --prune

# Export current environment
conda env export --name tracking-barbell-exercises > environment.yml

# Remove environment
conda env remove --name tracking-barbell-exercises

# List environments
conda env list
```

## 📈 Usage

### 1. Data Processing
```python
# Load and process raw sensor data
python src/data/make_dataset.py
```

### 2. Feature Engineering
```python
# Build features from processed data
python src/features/build_features.py
```

### 3. Train Models
```python
# Train classification models
python src/models/train_model.py
```

### Using Notebooks
```bash
jupyter notebook
# Open train_model.ipynb or remove_outliers.ipynb
```

## 🔬 Feature Engineering Pipeline

### 1. **Data Preprocessing**
   - Parse filenames to extract metadata (participant, exercise, intensity)
   - Merge accelerometer and gyroscope data
   - Handle missing values with interpolation

### 2. **Outlier Detection**
   - Chauvenet's criterion for outlier removal
   - Statistical analysis of sensor readings

### 3. **Signal Processing**
   - Butterworth low-pass filter (cutoff: 1.2 Hz)
   - Noise reduction and smoothing

### 4. **Temporal Features**
   - Rolling statistics (mean, std, min, max)
   - Set duration calculations
   - Window-based aggregations

### 5. **Frequency Features**
   - Fourier transformation
   - Power spectral density
   - Dominant frequency extraction

### 6. **Dimensionality Reduction**
   - Principal Component Analysis (PCA)
   - 3 principal components capturing variance

### 7. **Clustering**
   - K-Means clustering for pattern recognition
   - Additional cluster-based features

## 🤖 Machine Learning Models

The project implements multiple classification algorithms:
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Neural Networks

### Model Selection Process
1. Forward feature selection (top 10 features)
2. Grid search for hyperparameter tuning
3. Cross-validation for robustness
4. Performance evaluation with confusion matrices

### Feature Sets
- **Set 1**: Basic sensor features (6 features)
- **Set 2**: Basic + square + PCA (9 features)
- **Set 3**: Set 2 + temporal features
- **Set 4**: Set 3 + frequency + cluster features (full feature set)

## 📊 Results

The system achieves high accuracy in classifying barbell exercises using engineered features from raw sensor data. Key features identified include:
- PCA components
- Exercise duration
- Frequency domain features
- Temporal aggregations
- Magnitude calculations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Real-time prediction capabilities
- Mobile app integration
- Additional exercise types
- Deep learning models (LSTM, CNN)
- Deployment pipeline

## 📄 License

This project is available for educational and research purposes.

## 👥 Authors

- Your Name/Team

## 🙏 Acknowledgments

- MetaMotion sensor platform for data collection
- Exercise participants for dataset creation
- Open-source machine learning community

---

**Note**: This project is designed for educational purposes in the field of sports analytics and wearable sensor applications.