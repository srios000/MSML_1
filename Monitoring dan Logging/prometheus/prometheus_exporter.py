#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Prometheus Exporter
--------------------------------
Script that collects additional metrics for the Wine Quality prediction model
and exports them to Prometheus
"""

import os
import time
import logging
import argparse
import psutil
import requests
import threading
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Summary, Histogram

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("exporter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_exporter")

# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage percentage')
PREDICTION_RATE = Gauge('api_prediction_rate', 'Prediction requests per minute')

# API health metrics
API_UP = Gauge('api_up', 'API health status (1=up, 0=down)')
API_LATENCY = Gauge('api_latency_seconds', 'API response latency in seconds')

# Data drift metrics
FEATURE_DRIFT = Gauge('model_feature_drift', 'Feature drift score', ['feature'])
PREDICTION_DRIFT = Gauge('model_prediction_drift', 'Prediction distribution drift')
AVERAGE_PREDICTION = Gauge('model_average_prediction', 'Moving average of prediction values')
PREDICTION_STDDEV = Gauge('model_prediction_stddev', 'Standard deviation of recent predictions')

# Request metrics
REQUEST_COUNT = Counter('api_request_count_total', 'Total number of API requests', ['endpoint', 'method', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration in seconds',
                             ['endpoint'], buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))

class ModelMonitor:
    def __init__(self, api_url="http://localhost:5000", exporter_port=9092):
        self.api_url = api_url
        self.exporter_port = exporter_port
        self.running = False
        self.prediction_history = []
        self.feature_history = {}
        self.reference_data = None
        
        logger.info(f"Initializing ModelMonitor with API URL: {api_url}, exporter port: {exporter_port}")

    def check_api_health(self):
        """Check if the API is up and measure latency"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            latency = time.time() - start_time
            
            API_UP.set(1 if response.status_code == 200 else 0)
            API_LATENCY.set(latency)
            
            REQUEST_COUNT.labels(endpoint='/health', method='GET', status=response.status_code).inc()
            
            logger.info(f"API health check: status={response.status_code}, latency={latency:.4f}s")
            return True
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            API_UP.set(0)
            return False

    def get_model_info(self):
        """Get model information from the API"""
        try:
            response = requests.get(f"{self.api_url}/model/info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"Model info retrieved: {model_info['model_type']}")
                return model_info
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None

    def monitor_system_metrics(self):
        """Monitor system metrics (CPU, memory, disk)"""
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage('/').percent)
        
        logger.debug(f"System metrics: CPU={psutil.cpu_percent()}%, Memory={psutil.virtual_memory().percent}%, Disk={psutil.disk_usage('/').percent}%")

    def make_test_prediction(self):
        """Make a test prediction to monitor model behavior"""
        test_wine = {
            "fixed_acidity": 7.0,
            "volatile_acidity": 0.4,
            "citric_acid": 0.3,
            "residual_sugar": 5.0,
            "chlorides": 0.05,
            "free_sulfur_dioxide": 30.0,
            "total_sulfur_dioxide": 100.0,
            "density": 0.995,
            "pH": 3.2,
            "sulphates": 0.5,
            "alcohol": 10.0
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.api_url}/predict", json=test_wine, timeout=5)
            latency = time.time() - start_time
            
            REQUEST_DURATION.labels(endpoint='/predict').observe(latency)
            REQUEST_COUNT.labels(endpoint='/predict', method='POST', status=response.status_code).inc()
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["quality_prediction"]
                
                self.prediction_history.append(prediction)
                if len(self.prediction_history) > 100:
                    self.prediction_history.pop(0)
                
                AVERAGE_PREDICTION.set(np.mean(self.prediction_history))
                if len(self.prediction_history) > 1:
                    PREDICTION_STDDEV.set(np.std(self.prediction_history))
                
                for feature, value in test_wine.items():
                    if feature not in self.feature_history:
                        self.feature_history[feature] = []
                    
                    self.feature_history[feature].append(value)
                    if len(self.feature_history[feature]) > 100:
                        self.feature_history[feature].pop(0)
                
                logger.info(f"Test prediction: {prediction}, latency: {latency:.4f}s")
                return prediction
            else:
                logger.error(f"Test prediction failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error making test prediction: {str(e)}")
            return None

    def detect_feature_drift(self):
        """Detect drift in feature distribution"""
        if not self.feature_history:
            return
        
        for feature, values in self.feature_history.items():
            if len(values) <= 1:
                continue
                
            historical_mean = np.mean(values[:-1]) if len(values) > 1 else values[0]
            historical_std = np.std(values[:-1]) if len(values) > 1 else 1.0
            
            if historical_std == 0:
                historical_std = 1.0
                
            latest = values[-1]
            z_score = abs((latest - historical_mean) / historical_std)
            
            FEATURE_DRIFT.labels(feature=feature).set(z_score)
            
            if z_score > 2.0:
                logger.warning(f"Feature drift detected for {feature}: z-score={z_score:.2f}")

    def detect_prediction_drift(self):
        """Detect drift in prediction distribution"""
        if len(self.prediction_history) <= 1:
            return
            
        if len(self.prediction_history) >= 10:
            recent = self.prediction_history[-10:]
            overall = self.prediction_history
            
            recent_mean = np.mean(recent)
            overall_mean = np.mean(overall)
            overall_std = np.std(overall)
            
            if overall_std == 0:
                overall_std = 1.0
                
            drift_score = abs((recent_mean - overall_mean) / overall_std)
            
            PREDICTION_DRIFT.set(drift_score)
            
            if drift_score > 2.0:
                logger.warning(f"Prediction drift detected: score={drift_score:.2f}")

    def calculate_prediction_rate(self):
        """Calculate prediction rate by monitoring request count changes"""
        try:
            response = requests.get("http://localhost:8000/metrics")
            if response.status_code == 200:
                lines = response.text.split('\n')
                for line in lines:
                    if 'wine_prediction_total' in line and not line.startswith('#'):
                        current_count = float(line.split()[1])
                        
                        if not hasattr(self, 'last_count_check'):
                            self.last_count_check = {"time": time.time(), "count": current_count}
                        else:
                            now = time.time()
                            time_diff = now - self.last_count_check["time"]
                            count_diff = current_count - self.last_count_check["count"]
                            
                            if time_diff > 0:
                                rate = (count_diff / time_diff) * 60
                                PREDICTION_RATE.set(rate)
                                
                                self.last_count_check = {"time": now, "count": current_count}
                                
                                logger.debug(f"Prediction rate: {rate:.2f} per minute")
        except Exception as e:
            logger.error(f"Error calculating prediction rate: {str(e)}")

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        self.running = True
        
        while self.running:
            try:
                is_healthy = self.check_api_health()
                
                if is_healthy:
                    self.monitor_system_metrics()
                    
                    self.calculate_prediction_rate()
                    
                    if int(time.time()) % 10 == 0:
                        self.make_test_prediction()
                        self.detect_feature_drift()
                        self.detect_prediction_drift()
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)

    def start(self):
        """Start the monitoring exporter"""
        start_http_server(self.exporter_port)
        logger.info(f"Started Prometheus metrics server on port {self.exporter_port}")
        
        threading.Thread(target=self.run_monitoring_loop, daemon=True).start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping monitoring exporter")
            self.running = False

def main():
    parser = argparse.ArgumentParser(description="Wine Quality Model Monitoring Exporter")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000",
                        help="URL of the Wine Quality API")
    parser.add_argument("--port", type=int, default=9092,
                        help="Port to expose Prometheus metrics")
    
    args = parser.parse_args()
    
    monitor = ModelMonitor(api_url=args.api_url, exporter_port=args.port)
    monitor.start()

if __name__ == "__main__":
    main()