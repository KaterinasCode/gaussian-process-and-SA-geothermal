import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gpflow

k1=gpflow.kernels.Matern12()
k2=gpflow.kernels.Matern32()
k3=gpflow.kernels.Matern52()
k4 = gpflow.kernels.RBF()
k5=gpflow.kernels.Linear()
k6=gpflow.kernels.Constant()
k7=gpflow.kernels.Cosine()
k8 = gpflow.kernels.White()
kernels = [k1,k2,k3,k4,k5,k1*k1, k2*k5, k4*k5, k7]

def plotmodel(X_test, y_test, y_pred):
    predicted_values_reshaped = y_pred.numpy().flatten()
    plt.scatter(X_test[:, 5], y_test, color='blue', label='Training Samples')
    plt.scatter(X_test[:, 5], predicted_values_reshaped, color='red', label='Predictions')
    plt.xlabel('First Input Dimension')
    plt.ylabel('Target Values')
    plt.title('Predictions vs Training Samples')
    plt.show()

# Initialize an array to store the mean squared errors for each fold
errors = []
logmarg = []
k = 3
X = sample_s

# Perform k-fold cross-validation
def cross_validation(kernel, X):
    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for the current fold
        X_train, X_test = sample_s[train_index], sample_s[test_index]
        y_train, y_test = obj2[train_index], obj2[test_index]
        data = (X_train.reshape(-1, 17), y_train.reshape(-1, 1))
        model = gpflow.models.GPR(data, kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        marginal_likelihood = model.log_marginal_likelihood()
        logmarg.append(marginal_likelihood.numpy())
        print("marginal likelihood",marginal_likelihood.numpy())
    # Make predictions on the test data
        y_pred = model.predict_y(X_test)[0]
        plotmodel(X_test, y_test, y_pred)
    # Calculate the mean squared error for the current fold
        error = mean_squared_error(y_test, y_pred)
        errors.append(error)
        print(model.trainable_parameters)
    return np.mean(errors), np.mean(logmarg)

cross_validation(kernel,X)
