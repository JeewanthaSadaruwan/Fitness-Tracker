# build_features.py
# Feature engineering and dimensionality reduction for MetaMotion sensor data

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'metamotion_processed.csv')
FEATURES_DATA_DIR = os.path.join('data', 'processed')

def extract_statistical_features(df, sensor_prefix):
	"""Extract mean, std, min, max, median for each axis of a sensor."""
	features = {}
	axes = ['X', 'Y', 'Z']
	for axis in axes:
		col = f'{axis}_{sensor_prefix}' if f'{axis}_{sensor_prefix}' in df.columns else f'{axis}'
		if col in df.columns:
			features[f'{sensor_prefix}_{axis}_mean'] = df[col].mean()
			features[f'{sensor_prefix}_{axis}_std'] = df[col].std()
			features[f'{sensor_prefix}_{axis}_min'] = df[col].min()
			features[f'{sensor_prefix}_{axis}_max'] = df[col].max()
			features[f'{sensor_prefix}_{axis}_median'] = df[col].median()
	return features

def extract_features(df):
	"""Extract features for each exercise segment."""
	feature_rows = []
	grouped = df.groupby('exercise')
	for label, group in grouped:
		row = {'exercise': label}
		# Statistical features for accelerometer and gyroscope
		row.update(extract_statistical_features(group, 'accel'))
		row.update(extract_statistical_features(group, 'gyro'))
		# Add more features as needed (e.g., energy, peak-to-peak)
		feature_rows.append(row)
	features_df = pd.DataFrame(feature_rows)
	return features_df

def apply_pca(features_df, n_components=5):
	"""Apply PCA to reduce feature dimensionality."""
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(features_df.drop('exercise', axis=1))
	pca = PCA(n_components=n_components)
	X_pca = pca.fit_transform(X_scaled)
	pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
	pca_df['exercise'] = features_df['exercise'].values
	return pca_df, pca

def save_features(df, filename):
	os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
	out_path = os.path.join(FEATURES_DATA_DIR, filename)
	df.to_csv(out_path, index=False)
	print(f"Saved features to {out_path}")

def main():
	print("Loading processed data...")
	df = pd.read_csv(PROCESSED_DATA_PATH)

	print("Extracting features...")
	features_df = extract_features(df)
	save_features(features_df, 'metamotion_features.csv')

	print("Applying PCA...")
	pca_df, pca = apply_pca(features_df)
	save_features(pca_df, 'metamotion_features_pca.csv')

	print("Feature engineering complete.")

if __name__ == "__main__":
	main()
