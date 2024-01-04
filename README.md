# Convolutional Neural Network with Cifar-10 dataset
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the Cifar10 dataset available in tensorflow.

## Requirements

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy
- Pandas

## Quick Start

1. Install dependencies:

```bash
pip install tensorflow matplotlib numpy pandas
```
2. Clone the repository:
```bash
git clone https://github.com/your-username/Cifar10-ConvNet.git
cd Cifar10-ConvNet
```

3. Run the script:
```bash
python cifar10_convnet.py
```
4. Check out the training and validation curves and the evaluation of test dataset

## What I Did

- Preprocessed Cifar10 images: normalized pixel values to the range [0, 1].
- Shuffled and split the training dataset into training and validation sets.
- Built a CNN with Batch Normalization, Convolutional, MaxPooling, and Dense layers.
- Added Dropout for regularization to prevent overfitting.
- Trained the model using the training set, validated on the validation set.
- Evaluated the final model on the test set.

## Model Architecture
- Input Layer: Batch Normalization
- 3 blocks of Convolutional Layers with MaxPooling
- Fully Connected Layers with ReLU activation
- Dropout Layer for regularization
- Output Layer with Softmax activation

## Results
- Achieved a test accuracy of 75%.
- Visualized loss and accuracy curves for training and validation sets.

## License
This project is licensed under the [MIT License](LICENSE).
