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

[Your content for the Training and Results section goes here. You might want to include information about the dataset used, training process, evaluation metrics, and any notable results or insights gained from the model's performance.]

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
