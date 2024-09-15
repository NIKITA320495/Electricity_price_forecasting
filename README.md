# Time Series Forecasting with Hybrid Model

## Overview

This project aims to build a hybrid machine learning model that combines time series data with external features to forecast future values. The model uses a Long Short-Term Memory (LSTM) network for processing time series data and Dense layers for handling additional features, culminating in a combined neural network that predicts the target variable.

## Project Structure

- `data/`: Contains the datasets used for training and testing.
- `notebooks/`: Jupyter notebooks with data exploration, preprocessing, and model training scripts.
- `src/`: Source code for data preprocessing, model definition, training, and evaluation.
- `requirements.txt`: List of required Python packages.
- `README.md`: This README file.

## Requirements

- Python 3.12
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Data

The dataset consists of time series data for electricity prices along with additional external features that may influence the prices. The primary data column used for forecasting is `price actual`.

## Model Architecture

The model is a hybrid neural network combining LSTM and Dense layers:
- **Inputs**: Two inputs - one for time series data and one for external features.
- **LSTM Layer**: Processes the time series input to capture temporal patterns.
- **Dense Layer**: Handles the external features.
- **Concatenation**: Combines the outputs of the LSTM and Dense layers.
- **Dense Layers**: Further processes the combined features to produce the final prediction.

## Training

The model is trained using:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Squared Error (MSE)

Run the training script in `src/train_model.py` to train the model on the dataset. The model will be trained for 100 epochs, with both training and validation performance reported.

## Usage

To preprocess the data, train the model, and evaluate its performance, follow these steps:

1. **Preprocess Data**: Use `src/preprocess_data.py` to prepare the dataset.
2. **Train Model**: Execute `src/train_model.py` to train the model.
3. **Evaluate Model**: Review the model's performance in the training logs.

## Results

After 100 epochs of training, the model achieved a training loss and MSE of 5.8989, and a validation loss and MSE of 5.0717. The lower validation metrics suggest that the model generalizes well to unseen data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras documentation for model development.
- Various open-source tools and libraries used in data processing and machine learning.

---

Feel free to adjust the details according to your project specifics and folder structure.
