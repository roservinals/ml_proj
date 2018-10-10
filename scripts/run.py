import proj1_helpers as helper
import datetime as datetime
import numpy as np
import ls as ls
import functions as func

yb, input_data, ids = helper.load_csv_data(data_path='train.csv',sub_sample=False)

y, tx = func.build_model_data(input_data, yb)

print('X shape')
print(tx.shape)

print('Y shape')
print(y.shape)

start_time = datetime.datetime.now()
w,e = ls.least_squares(y, tx)
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print('-----------Option 1: LS-----------')
print("LS: execution time={t:.3f} seconds".format(t=exection_time))
print("Error={err:.6f}".format(err=e))

########### GRADIENT DESCENT ###########
print('-----------Option 2: Gradient descent-----------')
# Define the parameters of the algorithm.
max_iters = 50
gamma = 0.7
# Initialization

w_initial = np.zeros(tx.shape[1])
# Start gradient descent.
start_time = datetime.datetime.now()
gradient_losses, gradient_ws = ls.least_squares_GD(y, tx, w_initial, max_iters, gamma)
end_time = datetime.datetime.now()
e2 = gradient_losses[-1]
# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
print("Error={err:.6f}".format(err=e2))

########### STOCHAIS GRADIENT DESCENT ###########
print('-----------Option 3: Stochastic Gradient descent-----------')
# Define the parameters of the algorithm.
max_iters = 50
gamma = 0.7
batch_size = 1

# Initialization
w_initial = np.zeros(tx.shape[1])

# Start SGD.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = ls.least_squares_SGD(
    y, tx, w_initial, batch_size, max_iters, gamma)
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
e3 = sgd_losses[-1]
print("Error={err:.6f}".format(err=e3))