# Electricity Price Forecasting with a Hybrid Neural Network

## Overview

This project develops a hybrid machine learning model to forecast electricity prices by combining time series data with external features. The core of the model leverages a Long Short-Term Memory (LSTM) network to capture temporal patterns in the data, while additional external features are processed using Dense layers. The combined neural network model aims to deliver accurate predictions of future electricity prices based on historical trends and influencing factors.

## Data

The dataset includes time series data for electricity prices, specifically focusing on the `price actual` column as the target variable. It also contains external factors, such as energy generation metrics (biomass, fossil fuels, wind, solar, etc.), which may impact price fluctuations. This multi-dimensional dataset allows the model to learn both temporal dependencies and relationships between various energy generation sources.

## Model Architecture

The model employs a hybrid approach, integrating both LSTM and Dense layers:
- **Inputs**:
  - **Time Series Input**: Captures the historical electricity price trends through LSTM layers.
  - **External Features Input**: Processes non-time-series features (e.g., energy generation data) using Dense layers.
- **LSTM Layer**: Learns temporal patterns in the electricity price data.
- **Dense Layer**: Learns patterns from the external features influencing electricity prices.
- **Concatenation**: Combines the LSTM output with external feature data to create a comprehensive feature set.
- **Dense Layers**: Additional layers refine and learn deeper representations from the combined features before outputting the final price prediction.

## Training

The model is trained using:
- **Optimizer**: Adam, which is well-suited for time series data and adaptive learning.
- **Loss Function**: Mean Squared Error (MSE), a standard metric for regression tasks.
- **Metrics**: Mean Squared Error (MSE) to evaluate the model’s performance during training and validation.

## Usage

To use this model:
1. **Data Preprocessing**: Preprocess the dataset, including time series transformation and handling of external features.
2. **Training the Model**: Train the hybrid model by running the provided training script. The model will use the preprocessed data to learn relationships between the time series and external features.
3. **Evaluation**: After training, the model’s performance on validation data will be assessed using the MSE.

## Results

After 100 epochs of training, the model achieved a training loss and MSE of **5.8989** and a validation loss and MSE of **5.0717**. The lower validation metrics indicate that the model generalizes well to unseen data and does not overfit the training data. This performance highlights the model’s ability to capture the underlying patterns in both time series data and external features, resulting in improved electricity price predictions.

## Conclusion

This hybrid model successfully integrates LSTM for time series forecasting with Dense layers for handling external features, demonstrating an effective approach to electricity price prediction. The model’s performance suggests that considering both historical trends and external factors such as energy generation can significantly improve forecasting accuracy.
