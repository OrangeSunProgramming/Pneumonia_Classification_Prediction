# Pneumonia Classification using Inception-like Neural Network

## Overview

This repository showcases a deep learning project for binary classification of chest X-rays to determine whether they indicate 'pneumonia' or 'normal'. The model is inspired by Google's Inception architecture, leveraging inception-like modules to efficiently capture a wide range of features from the X-ray images.

## Project Structure

- **/data/**: Contains example images of pneumonia and normal chest X-rays.
- **/src/**: Contains the code for the project:
  - `accuracy_metrics.py`: Calculates and displays precision, recall, F1-score, and accuracy metrics.
  - `chest_xray_model_setup.py`: Defines and sets up the inception-like neural network model.
  - `chest_xray_visual_gen.py`: Generates and visualizes training graphs and other related visualizations.
  - `prediction.py`: Handles model predictions and evaluations on test data.
- **/results/**: Includes saved model summaries, training graphs, and accuracy scores.
- `README.md`: This file, providing an overview and instructions.
- `requirements.txt`: Lists dependencies for setting up the project environment.
- `LICENSE`: Apache License 2.0 for project usage and distribution.

## Model Architecture

The model is based on an inception-like architecture with the following features:

- **Initial Layers**: Convolutional layers with varying kernel sizes (7x7, 1x1, 3x3) for feature extraction.
- **Inception-like Modules**: Each module includes 1x1, 3x3, and 5x5 convolutional layers along with max pooling, allowing the model to capture multi-scale features.
- **Global Average Pooling**: Reduces the spatial dimensions of the feature maps.
- **Dense Layer**: A final dense layer with a sigmoid activation function for binary classification.

## Training and Results

### Dataset

The dataset used for this project consists of chest X-ray images classified into two categories: 'pneumonia' and 'normal'. The images were split into training and validation sets, with 80% used for training and 20% for validation.

### Training Process

The model was trained using an inception-like architecture inspired by Google's Inception network. Training was performed for up to 100 epochs, but due to the EarlyStopping callback with a patience of 1, training stopped at epoch 7. The callback monitored the validation loss (`val_loss`) and restored the best weights to prevent overfitting.

The training process included:

- **Optimizer**: Adam with a learning rate of 5e-5.
- **Loss Function**: Binary Cross-Entropy.
- **Metrics**: Accuracy, Precision, Recall, F1-score.

### Evaluation Metrics

After training, the model's performance was evaluated on the validation set. The following metrics were achieved:

- **Precision**: 0.9766
- **Recall**: 0.9790
- **F1-score**: 0.9778
- **Accuracy**: 0.9771
- **Loss**: 0.0765

These metrics indicate that the model performs exceptionally well in distinguishing between pneumonia and normal X-rays. The high precision and recall values suggest that the model is effective at correctly identifying both classes with a low rate of false positives and false negatives.

### Notable Results and Insights

- **Early Stopping**: The use of EarlyStopping with patience 1 and monitoring `val_loss` was effective in preventing overfitting, as indicated by the performance metrics.
- **High Performance**: The model achieved a high accuracy of 97.71% and very strong precision and recall scores, demonstrating its robustness and reliability in classifying chest X-ray images.

Overall, the results demonstrate that the inception-like model is highly capable of performing binary classification on chest X-ray images, achieving both high accuracy and balanced performance across key metrics.


## Getting Started

1. Clone this repository:

  - git clone https://github.com/OrangeSunProgramming/pneumonia-classification.git
  - cd pneumonia-classification


2. Create a virtual environment and install dependencies:

  - python -m venv env
  - source env/bin/activate  # On Windows use `env\Scripts\activate`
  - pip install -r requirements.txt

## Run the Code

1. **Set up the model**:
   - Run `chest_xray_model_setup.py` to define and initialize the inception-like neural network model.

2. **Train the model and visualize the results**:
   - Use `chest_xray_visual_gen.py` to train the model and generate visualizations, including training graphs.

3. **Evaluate and test the model**:
   - Evaluate the model's performance with `accuracy_metrics.py` to get precision, recall, F1-score, and accuracy.
   - Use `prediction.py` to make predictions and test the model on new data.

## Results and Visualizations

The results of the model training, including accuracy scores and training graphs, are available in the `/results/` directory. Sample images from the dataset can be found in the `/data/` directory.

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0). See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or further information, please feel free to contact me at marcosmasipcompany@gmail.com.
