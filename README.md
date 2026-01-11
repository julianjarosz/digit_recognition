This repository builds a handwritten digit recognizer using a custom NumPy MLP and compares it with scikit-learn’s MLP.
Model: A multi-layer perceptron implemented in models.mlp_digit.MLP with ReLU hidden layers, softmax output, cross-entropy loss, Kaiming initialization, and mini-batch SGD.
Training: train.ipynb loads train.csv, normalizes inputs, performs a simple grid search over hidden layers, learning rate, epochs, and batch size, trains the custom MLP, evaluates on a validation split, prints a confusion matrix, and saves the best model.
Evaluation: evaluate.ipynb loads the saved model, predicts on test.csv, and writes a Kaggle-ready submission to submission.csv.
Comparison: compare_scikit.ipynb trains scikit-learn’s MLPClassifier with matched hyperparameters to benchmark against our implementation.
