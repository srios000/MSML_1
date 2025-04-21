#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Inference Script
-----------------------------
Script for making predictions using the deployed Wine Quality model API
"""

import argparse
import requests
import json
import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_inference")

def load_sample_data(file_path=None):
    """
    Load sample data for prediction
    """
    if file_path:
        try:
            # Try to load from the provided file
            logger.info(f"Loading sample data from {file_path}")
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
            
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    else:
        logger.info("Generating synthetic wine data")
        np.random.seed(42)
        
        data = pd.DataFrame({
            'fixed_acidity': np.random.uniform(4.0, 15.0, 10),
            'volatile_acidity': np.random.uniform(0.1, 1.2, 10),
            'citric_acid': np.random.uniform(0.0, 1.0, 10),
            'residual_sugar': np.random.uniform(0.9, 15.0, 10),
            'chlorides': np.random.uniform(0.01, 0.25, 10),
            'free_sulfur_dioxide': np.random.uniform(1.0, 72.0, 10),
            'total_sulfur_dioxide': np.random.uniform(6.0, 289.0, 10),
            'density': np.random.uniform(0.98, 1.01, 10),
            'pH': np.random.uniform(2.7, 4.0, 10),
            'sulphates': np.random.uniform(0.2, 2.0, 10),
            'alcohol': np.random.uniform(8.0, 14.0, 10)
        })
        
        logger.info(f"Generated {len(data)} synthetic samples")
        return data

def make_prediction(api_url, data):
    """
    Make predictions using the API
    """
    if isinstance(data, pd.DataFrame):
        if len(data) > 1:
            wine_list = data.to_dict('records')
            payload = {"wines": wine_list}
            
            logger.info(f"Making batch prediction for {len(wine_list)} samples")
            
            try:
                start_time = time.time()
                response = requests.post(f"{api_url}/predict/batch", json=payload, timeout=10)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Batch prediction successful. Latency: {latency:.4f}s")
                    return result, latency
                else:
                    logger.error(f"Batch prediction failed: {response.status_code}, {response.text}")
                    return None, latency
            except Exception as e:
                logger.error(f"Error making batch prediction: {str(e)}")
                return None, 0
        
        else:
            wine_data = data.iloc[0].to_dict()
            logger.info(f"Making single prediction for sample: {wine_data}")
            
            try:
                start_time = time.time()
                response = requests.post(f"{api_url}/predict", json=wine_data, timeout=5)
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Prediction successful. Result: {result['quality_prediction']}, Latency: {latency:.4f}s")
                    return result, latency
                else:
                    logger.error(f"Prediction failed: {response.status_code}, {response.text}")
                    return None, latency
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                return None, 0
    else:
        logger.error("Data must be a pandas DataFrame")
        return None, 0

def run_benchmark(api_url, num_requests=100, batch_size=10):
    """
    Run a benchmark to test API performance
    """
    logger.info(f"Running benchmark with {num_requests} requests (batch size: {batch_size})")
    
    np.random.seed(42)
    all_data = pd.DataFrame({
        'fixed_acidity': np.random.uniform(4.0, 15.0, num_requests),
        'volatile_acidity': np.random.uniform(0.1, 1.2, num_requests),
        'citric_acid': np.random.uniform(0.0, 1.0, num_requests),
        'residual_sugar': np.random.uniform(0.9, 15.0, num_requests),
        'chlorides': np.random.uniform(0.01, 0.25, num_requests),
        'free_sulfur_dioxide': np.random.uniform(1.0, 72.0, num_requests),
        'total_sulfur_dioxide': np.random.uniform(6.0, 289.0, num_requests),
        'density': np.random.uniform(0.98, 1.01, num_requests),
        'pH': np.random.uniform(2.7, 4.0, num_requests),
        'sulphates': np.random.uniform(0.2, 2.0, num_requests),
        'alcohol': np.random.uniform(8.0, 14.0, num_requests)
    })
    
    latencies = []
    predictions = []
    success_count = 0
    
    start_time = time.time()
    
    if batch_size > 1:
        for i in range(0, num_requests, batch_size):
            end_idx = min(i + batch_size, num_requests)
            batch_data = all_data.iloc[i:end_idx]
            
            result, latency = make_prediction(api_url, batch_data)
            latencies.append(latency)
            
            if result:
                success_count += len(result["quality_predictions"])
                predictions.extend(result["quality_predictions"])
    else:
        for i in range(num_requests):
            single_data = all_data.iloc[[i]]
            
            result, latency = make_prediction(api_url, single_data)
            latencies.append(latency)
            
            if result:
                success_count += 1
                predictions.append(result["quality_prediction"])
    
    total_time = time.time() - start_time
    
    logger.info(f"Benchmark completed in {total_time:.2f} seconds")
    logger.info(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.2f}%)")
    
    if latencies:
        logger.info(f"Average latency: {np.mean(latencies):.4f}s")
        logger.info(f"Min latency: {np.min(latencies):.4f}s")
        logger.info(f"Max latency: {np.max(latencies):.4f}s")
        logger.info(f"P95 latency: {np.percentile(latencies, 95):.4f}s")
    
    if predictions:
        logger.info(f"Average prediction: {np.mean(predictions):.2f}")
        logger.info(f"Min prediction: {np.min(predictions):.2f}")
        logger.info(f"Max prediction: {np.max(predictions):.2f}")
    
    if latencies:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(latencies, kde=True)
        plt.title("API Latency Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Count")
        
        if predictions:
            plt.subplot(1, 2, 2)
            sns.histplot(predictions, kde=True, bins=10)
            plt.title("Prediction Distribution")
            plt.xlabel("Wine Quality Prediction")
            plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        logger.info("Saved benchmark visualization to benchmark_results.png")
    
    return {
        "total_time": total_time,
        "success_rate": success_count/num_requests,
        "avg_latency": np.mean(latencies) if latencies else None,
        "predictions": predictions
    }

def main():
    parser = argparse.ArgumentParser(description="Wine Quality Model Inference")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000",
                        help="URL of the Wine Quality API")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to file with input data (CSV or JSON)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run a benchmark test")
    parser.add_argument("--num_requests", type=int, default=100,
                        help="Number of requests for benchmark")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for benchmark")
    
    args = parser.parse_args()
    
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"API is healthy: {response.json()}")
        else:
            logger.error(f"API health check failed: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"Error connecting to API: {str(e)}")
        return
    
    if args.benchmark:
        run_benchmark(args.api_url, args.num_requests, args.batch_size)
        return
    
    data = load_sample_data(args.file)
    if data is not None:
        make_prediction(args.api_url, data)

if __name__ == "__main__":
    main()