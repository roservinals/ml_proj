import numpy as np
import costs as costs
import functions as func

def least_squares(y, tx):
    """calculate the least squares solution."""
    w=np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e=costs.compute_ls(y-tx.dot(w))
    return w,e

def compute_gradient(y, tx, w): #same mse
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        gr, e = compute_gradient(y, tx, w)
        loss = costs.compute_ls(e)
        # gradient w by descent update
        w = w - gamma * gr
        # store w and loss
        ws.append(w)
        losses.append(loss)
       # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in func.batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = costs.compute_ls(e)
            # store w and loss
            ws.append(w)
            losses.append(loss)

       # print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws