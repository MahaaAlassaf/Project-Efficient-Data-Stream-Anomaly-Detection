## Overview
This project implements an anomaly detection system using simulated data streams. The system provides functionality for simulating data with noise, detecting anomalies, visualizing results, and running real-time data simulations.

## How to Run the Code
There are two ways to run the code in this project:
1. By using the Jupyter Notebook (`Algorithm.ipynb`).
2. By running the Python script (`Algorithm.py`) in a virtual environment using `requirements.txt`.

## Anomaly Detection System: Simulation, Detection, Visualization, and Real-Time Streaming

### Step 1: Simulating the Data Stream
The function is `generate_data_stream(size=1000, seasonality=50, noise_level=5, anomaly_rate=0.01)`.

- **size**: Indicates how many data points there are.
- **seasonality**: Regulates how frequently the data's seasonal trends occur.
- **noise_level**: Increases noise to represent unpredictability in the real world.
- **anomaly_rate**: Ascertains the percentage of anomalous data points.

**Method**:
- **Seasonal Patterns**: Using sine functions, regular fluctuations are simulated.
- **Trend**: Added to the data to indicate a general upward or downward direction.
- **Noise**: Added to the data to make it less predictable and more realistic.

**Reasoning**:  
The anomaly detection system can be tested in a realistic setting by simulating data including seasonality, noise, and abnormalities. It aids in assessing the algorithm's performance under various circumstances.

### Step 2: Implementing the Anomaly Detection Algorithm

**Goal**: Implement an anomaly detection algorithm and apply it to the simulated data.

**Description**:  
Class: `RefinedAnomalyDetector`  
- **window_size**: The amount of recent data points considered for anomaly detection.  
- **ema_alpha**: The Exponential Moving Average (EMA) smoothing factor.  
- **z_threshold**: A Z-score threshold determining whether a data point is an anomaly.

**Method**:  
- **Z-Score Calculation**: Determines a data point's distance from the recent data window mean in standard deviations. Anomalies are indicated by points having a Z-score greater than a predetermined value.

**Reasoning**:  
The EMA and Z-score combination allows for detecting abnormalities in time-series data by identifying deviations from expected patterns. EMA smooths short-term variations, while the Z-score provides a standardized method to identify anomalies.

### Step 3: Visualization

**Objective**: Provide various visualizations to understand and compare the data, anomalies, and the performance of the anomaly detection algorithm.

**Description**:  
Function: `plot_comparison(data_stream, ema_points, anomaly_indices, anomaly_values)`

**Plots**:
- **Data Stream vs. EMA**: Visualizes the original data stream alongside the smoothed EMA.
- **Data Stream with Anomalies**: Highlights detected anomalies on the data stream.
- **Rolling Statistics**: Plots rolling mean and standard deviation to observe deviations.
- **Histogram of Data Distribution**: Shows the frequency distribution of data stream values.
- **Histogram of Anomalies**: Displays the distribution of detected anomaly values.
- **Anomalies Over Time**: Shows where anomalies occur over time in the data stream.

**Reasoning**:  
Multiple visualizations provide a comprehensive view of the data and the anomalies detected. This helps in understanding the effectiveness of the anomaly detection algorithm, identifying patterns, and comparing different aspects of the data.

### Step 4: Real-Time Data Stream Simulation

**Objective**: Simulate real-time data streaming and update visualizations dynamically.

**Description**:  
Function: `real_time_simulation(detector, data_stream, interval=10)`  
- **interval**: Frequency at which the plot is updated with new data points.

**Process**:
- **Real-Time Plotting**: Continuously updates the plot with new data points and detected anomalies.
- **Real-Time Anomalies**: Highlights anomalies as they are detected in real-time.

**Reasoning**:  
Real-time simulation mimics actual data streaming scenarios, allowing for the testing of the anomaly detection algorithm in a live context. This helps in evaluating the algorithm’s performance and responsiveness to continuous data input.

### Summary
- **Data Stream Simulation**: Creates a realistic dataset for testing.
- **Anomaly Detection**: Applies an EMA and Z-score based method to detect anomalies.
- **Visualization**: Provides multiple plots to understand data and detection results.
- **Real-Time Simulation**: Mimics live data streaming and updates plots in real-time.
