# Convolutional Neural Network with Cifar-10 dataset
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the Cifar10 dataset available in tensorflow.

## Requirements

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy
- Pandas
------
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
-----
## What I Did

- Preprocessed Cifar10 images: normalized pixel values to the range [0, 1].
- Shuffled and split the training dataset into training and validation sets.
- Built a CNN with Batch Normalization, Convolutional, MaxPooling, and Dense layers.
- Added Dropout for regularization to prevent overfitting.
- Trained the model using the training set, validated on the validation set.
- Evaluated the final model on the test set.
-----
## Data Handling and Model Optimization

### Data Challenges
Initially, data leakage posed a risk to model performance evident from the graph curves.

### Solutions Implemented
1. **Data Preprocessing:**
   - Normalized pixel values to [0, 1].

2. **Data Splitting and Shuffling:**
   - Shuffled and split the training dataset into training and validation sets.

3. **Model Enhancements:**
   - Added Batch Normalization for stable training.
   - Employed a Dropout layer to prevent overfitting.
  
-----
## Model Architecture
- Input Layer: Batch Normalization
- 3 Convolutional blocks with MaxPooling
- Fully Connected Layers with ReLU
- Dropout Layer for regularization
- Output: Softmax
------
## Results
- Achieved a test accuracy of 75%.
- Visualized loss and accuracy curves for training and validation sets.
------
## License
This project is licensed under the [MIT License](LICENSE).
