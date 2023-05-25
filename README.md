# Sign Language Recognition using Convolutional Neural Networks (CNN)

Description:
Developed a Sign Language Recognition system using Convolutional Neural Networks (CNN) to recognize hand gestures in sign language. The project involved the following key steps:

Data Preparation:

Downloaded and preprocessed the Sign Language MNIST dataset using pandas.
Split the dataset into training and testing sets.
Performed label binarization to convert categorical labels into binary vectors.
Data Augmentation:

Applied image data augmentation techniques using ImageDataGenerator from Keras.
Techniques included random rotation, zooming, shifting, and flipping to enhance the training dataset and improve model generalization.
Model Architecture:

Built a CNN model using Keras with sequential layers.
Utilized convolutional layers with different filter sizes and activation functions to extract local features and recognize patterns in the sign language images.
Employed batch normalization to normalize and scale input data.
Incorporated dropout layers for regularization to prevent overfitting.
Flattened the output and added fully connected dense layers with a softmax activation function for multi-class classification.
Model Training:

Compiled the model using the Adam optimizer and categorical cross-entropy loss function.
Trained the model using the augmented training data and evaluated its performance on the testing data.
Implemented a learning rate reduction callback to adaptively adjust the learning rate during training for improved convergence.
Evaluation and Visualization:

Generated a confusion matrix to evaluate the model's performance and visualize the prediction results.
Used seaborn and matplotlib libraries to create heatmaps and visualizations of the confusion matrix.
This Sign Language Recognition project demonstrates your proficiency in deep learning techniques, specifically in building CNN models for image classification tasks. It showcases your ability to preprocess data, apply data augmentation, design and train neural network models, and evaluate their performance.
