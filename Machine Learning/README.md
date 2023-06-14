# Capstone-Bangkit-2023-Journey

## Journey Hyperparameter Tuning
This repository contains code for hyperparameter tuning of a deep learning model using TensorFlow and Keras. The model is trained to predict positions based on various attributes, such as disability type, skills, and position. The hyperparameter tuning is performed using the Kerastuner library.

## Dataset
The dataset used for training the model is stored in the journey_dataset.csv file. It contains the following attributes: disability_type, skills_one, skills_two, and position. The dataset is split into a training set and a test set, with a 80:20 ratio.

## Model Architecture
The model architecture is defined in the build_model function. It consists of an embedding layer, LSTM layer, and a dense output layer with softmax activation. The hyperparameters for the model, such as embedding dimension, LSTM units, and learning rate, are tuned using random search with the Kerastuner library.

## Hyperparameter Tuning
The hyperparameter tuning is performed using the RandomSearch tuner. It searches for the best hyperparameters by evaluating multiple trials of the model with different hyperparameter configurations. The tuner searches for the optimal values of embedding dimension, LSTM units, and learning rate. The best hyperparameters are then used to build the final model.

## Training
The model is trained using the optimal hyperparameters obtained from the tuning process. The training is performed on the training set for a specified number of epochs. The model is compiled with the Adam optimizer and categorical cross-entropy loss. Early stopping is applied to prevent overfitting.

## Evaluation
After training, the model is evaluated on the test set to measure its performance. The test loss and accuracy are reported.

## Saving the Model
The trained model is saved in two formats:

1. trained_model1.h5: The model is saved in the HDF5 format using Keras' model.save() function.
2. model.tflite: The model is converted to the TensorFlow Lite format using the TensorFlow Lite Converter.

## Requirement
To run the code in this repository, the following dependencies are required:

1. TensorFlow
2. NumPy
3. scikit-learn
4. Keras Tuner

You can install the required packages by running the following command:

`pip install tensorflow numpy scikit-learn keras-tuner`

## Usage
To run the hyperparameter tuning and training process, execute the following command:

`python hyperparameter_tuning.py`

The trained model and the TensorFlow Lite model will be saved in the current directory.

Feel free to modify the hyperparameters, model architecture, or any other aspect of the code to suit your specific needs.

That's it! You can use this README.md file as a starting point and customize it further based on your requirements.
