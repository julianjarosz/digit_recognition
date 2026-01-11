# Handwritten Digit Recognizer (NumPy MLP vs scikit-learn)

This repository implements a handwritten digit recognizer from scratch using **NumPy**, and benchmarks it against **scikit-learn's MLPClassifier**.

---

## Model

- Custom **multi-layer perceptron (MLP)** implemented in `models/mlp_digit.py`
- **Hidden layers:** ReLU activations
- **Output layer:** Softmax
- **Loss function:** Cross-Entropy
- **Initialization:** Kaiming He
- **Optimizer:** Mini-batch Stochastic Gradient Descent (SGD)



## Training

1. Loads `train.csv`
2. Normalizes input features
3. **Grid search** over:
   - Number of hidden layers
   - Learning rate
   - Epochs
   - Batch size
4. Trains and validates on dataset split
5. Prints **confusion matrix**
6. Saves **best model**

---

## Evaluation

- Loads trained model
- Predicts `test.csv`
- Generates **Kaggle-ready** `submission.csv`

---

## Comparison

Benchmarks **scikit-learn's MLPClassifier** with equivalent hyperparameters:
- Accuracy metrics
- Training time
- Convergence comparison

---

## Requirements
Listed in `requirements.txt`
## Usage
In order to test it out, you need to get the Kaggle dataset and place it into the src/data folder. Then create a venv with requirements installed.
