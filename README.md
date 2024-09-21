🦠  Deep Neural Network for Malaria Infected Cell Recognition

📋 Problem Statement and Dataset
The dataset consists of 27,558 cell images with an equal number of parasitized and uninfected cells. A level-set based algorithm was applied to detect and segment the red blood cells. The images were collected and annotated by medical professionals.

The goal is to develop a Convolutional Neural Network (CNN) that can classify cells as parasitized or uninfected.



🧠 Neural Network Model
The neural network model is designed as a convolutional neural network (CNN) with the following architecture:

Input Layer: 130x130 RGB images
Conv2D Layers: To extract features from the images
Pooling Layers: To reduce the spatial dimensions
Flatten Layer: To convert 2D feature maps into a 1D vector
Dense Layer: Fully connected layer to make the final predictions
Output Layer: Single neuron with sigmoid activation for binary classification


🛠️ Design Steps
Import Libraries: TensorFlow, Keras, and preprocessing utilities.
Download and Load Dataset: Organize the dataset into training and testing folders.
Data Splitting: Split the dataset into training and testing sets.
Image Data Generation: Use augmentation techniques for better model generalization.
Build CNN Model: Define convolutional, pooling, and dense layers.
Train the Model: Fit the model to the training data.
Evaluate the Model: Plot training and validation loss over epochs.
Test on New Data: Make predictions using the trained model.
⚙️ Training and Performance Plots
Training Loss and Validation Loss Over Iterations:
The plot showcases how the model learns during training.


📊 Evaluation Metrics
After training the model, it is evaluated on a test set. The performance metrics, including classification report and confusion matrix, are generated.

Classification Report
The classification report provides detailed performance metrics like precision, recall, and F1 score.



Confusion Matrix
The confusion matrix shows how well the model is classifying parasitized and uninfected cells.



🔍 New Sample Data Prediction
Here is an example of the model predicting whether a new cell image is parasitized or uninfected:



🔮 Conclusion
Thus, a deep neural network for malaria-infected cell recognition was developed, and its performance was successfully analyzed. The trained model can predict malaria-infected cells with high accuracy, aiding in rapid and reliable diagnosis.

