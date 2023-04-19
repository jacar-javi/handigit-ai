# handigit-ai
Develop a neural network that can correctly classify images of handwritten digits. Training using the MNIST dataset.

##Step 1: Develop an Artificial Neural Network capable of classifying handwritten digits

We will use a Keras library based secuential Artificial Neural Network with 3 layers:

1. Flattening Layer: Converts 28x28 pixel 2D images to a 784 element 1D vector.
2. Dense Hidden Layer: With 128 neurons and ReLU (Rectified Linear Unit) activation function. This layer is responsible for learning the relevant features of the images for digit classification.
3. Dense Output Layer: With 10 neurons and Softmax activation function. This layer generates the probabilities that the input image belongs to each of the 10 possible classes (digits 0 to 9).

The model is compiled by specifying the optimizer (in this case 'adam'), the loss function (in this case 'categorical_crossentropy'), and the evaluation metric (in this case 'accuracy').


##Step 2: Train NN with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and evaluate its accuracy.

1. Load the MNIST dataset and split it into two subsets: training and testing. 
2. The pixel values of the images are normalized so that they are in the range of 0 to 1. 
3. Labels are converted to one-hot encoding so that they can be used in the neural network.

The model is trained using the training dataset (x_train, y_train) and validating it on the test dataset (x_test, y_test). Training is performed for a specified number of epochs (in this case, 10) and with a specified batch_size (in this case, 32).

Model performance during training is visualized using graphs showing accuracy and loss as a function of the number of epochs. These graphs allow you to assess how the model learns over time and detect potential overfitting or underfitting issues.

<img width="739" alt="model_training_epochs" src="https://user-images.githubusercontent.com/17501624/233156514-70ec0c53-937c-4255-907c-cc6453e96bb0.png">

<img width="775" alt="model_training_performance" src="https://user-images.githubusercontent.com/17501624/233157813-4455c1d6-0704-4b7a-bb2c-ffe8592073cd.png">


After model training is complete, the model's performance is evaluated on the test dataset for loss and accuracy.

<img width="204" alt="model_training_validation" src="https://user-images.githubusercontent.com/17501624/233157293-6eb792ec-f4ce-48f7-96b5-e997a8262254.png">




##Step 3: Save trained model to use in future apps (TODO)
