#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Data Preprocessing Automation Script
-------------------------------------------------
Script ini bertujuan untuk melakukan preprocessing data Wine Quality secara otomatis
dengan mengaplikasikan teknik-teknik yang telah dieksplorasi pada notebook.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from ucimlrepo import fetch_ucirepo
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_preprocessing")

def load_data():
    """
    Memuat dataset Wine Quality dari UCI ML Repository
    """
    logger.info("Memuat dataset Wine Quality dari UCI ML Repository...")
    try:
        wine_quality = fetch_ucirepo(id=186)
        
        # Data features (X) dan target (y)
        X = wine_quality.data.features
        y = wine_quality.data.targets
        
        # Menggabungkan data fitur dan target
        data = pd.concat([X, y], axis=1)
        
        logger.info(f"Dataset berhasil dimuat dengan dimensi: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Gagal memuat dataset: {str(e)}")
        raise

def handle_missing_values(data):
    """
    Melakukan penanganan missing values dengan imputasi median
    """
    logger.info("Memeriksa dan menangani missing values...")
    
    # Memeriksa missing values
    missing_count = data.isnull().sum().sum()
    
    if missing_count > 0:
        logger.info(f"Ditemukan {missing_count} missing values. Melakukan imputasi...")
        
        # Imputasi dengan median
        imputer = SimpleImputer(strategy='median')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        logger.info("Missing values telah diisi dengan nilai median")
        return data_imputed
    else:
        logger.info("Tidak ditemukan missing values pada dataset")
        return data.copy()

def handle_outliers(df, columns):
    """
    Melakukan deteksi dan penanganan outlier menggunakan metode IQR
    """
    logger.info("Mendeteksi dan menangani outliers...")
    
    df_clean = df.copy()
    outlier_count = 0
    
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outliers = df_clean[(df_clean[column] < lower_bound) | 
                                  (df_clean[column] > upper_bound)].shape[0]
        
        outlier_count += column_outliers
        
        logger.info(f"Jumlah outlier pada '{column}': {column_outliers}")
        
        # Capping outlier dengan batas bawah dan batas atas
        df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
        df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
    
    logger.info(f"Total outlier yang ditangani: {outlier_count}")
    return df_clean

def standardize_features(X):
    """
    Melakukan standardisasi fitur
    """
    logger.info("Melakukan standardisasi fitur...")
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    logger.info("Standardisasi fitur selesai")
    return X_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi training dan testing set
    """
    logger.info(f"Membagi data dengan test_size={test_size}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data berhasil dibagi. Train set: {X_train.shape[0]} sampel, Test set: {X_test.shape[0]} sampel")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir='winequality_preprocessing'):
    """
    Menyimpan data yang telah diproses
    """
    logger.info(f"Menyimpan data hasil preprocessing ke direktori '{output_dir}'...")
    
    # Membuat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Direktori '{output_dir}' telah dibuat")
    
    # Menggabungkan fitur dan target
    train_data = pd.concat([
        pd.DataFrame(X_train, columns=X_train.columns), 
        pd.DataFrame(y_train, columns=['quality'])
    ], axis=1)
    
    test_data = pd.concat([
        pd.DataFrame(X_test, columns=X_test.columns), 
        pd.DataFrame(y_test, columns=['quality'])
    ], axis=1)
    
    # Menyimpan data
    train_data.to_csv(f'{output_dir}/train_data.csv', index=False)
    test_data.to_csv(f'{output_dir}/test_data.csv', index=False)
    
    # Menyimpan scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    # Menyimpan informasi kolom
    with open(f'{output_dir}/feature_names.txt', 'w') as f:
        f.write(','.join(X_train.columns))
    
    logger.info(f"Data berhasil disimpan ke direktori '{output_dir}'")

def main():
    """
    Fungsi utama untuk menjalankan alur preprocessing
    """
    logger.info("Memulai proses preprocessing data Wine Quality...")
    
    try:
        # 1. Memuat data
        data = load_data()
        
        # 2. Penanganan missing values
        data_clean = handle_missing_values(data)
        
        # 3. Memisahkan fitur dan target
        X = data_clean.drop('quality', axis=1)
        y = data_clean['quality']
        
        # 4. Penanganan outlier pada fitur
        X_clean = handle_outliers(X, X.columns)
        
        # 5. Standardisasi fitur
        X_scaled, scaler = standardize_features(X_clean)
        
        # 6. Train-test split
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)
        
        # 7. Menyimpan data hasil preprocessing
        save_processed_data(X_train, X_test, y_train, y_test, scaler)
        
        logger.info("Preprocessing data Wine Quality selesai!")
        return True
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan dalam preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    main()