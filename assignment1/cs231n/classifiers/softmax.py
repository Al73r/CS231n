import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = X.shape[1]
  loss = 0.0
  dloss = 1.0
  dW = dloss * 0.5 * reg * 2 * W
  dloss = dloss * 1.0/num_train
  
  for i in range(num_train):
    scores = X[i].dot(W)
    idx_max = np.argmax(scores)
    max_score = scores[idx_max]
    scores -= max_score
    exp_scores = np.exp(scores)
    correct_exp_score = exp_scores[y[i]]
    sum_exp_scores = np.sum(exp_scores)
    rlt_div = correct_exp_score / sum_exp_scores
    loss_i = np.log(rlt_div)
    loss -= loss_i
    dloss_i = -1.0 * dloss
    drlt_div = dloss_i * 1.0/rlt_div
    dcorrect_exp_score = drlt_div * 1.0/sum_exp_scores
    dsum_exp_scores = drlt_div * -1.0 * correct_exp_score / (sum_exp_scores ** 2)
    dexp_scores = np.ones(exp_scores.shape) * dsum_exp_scores
    dexp_scores[y[i]] += dcorrect_exp_score
    dscores = dexp_scores * np.exp(scores)
    dmax_score = -1.0 * np.sum(dscores)
    dscores[idx_max] += dmax_score
    dW += X[i].reshape((num_dim, 1)).dot(dscores.reshape((1, num_classes)))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  idx_max = np.argmax(scores, 1)
  max_score = scores[range(num_train), idx_max]
  scores -= max_score.reshape((num_train, 1))
  exp_scores = np.exp(scores)
  correct_exp_score = exp_scores[range(num_train), y]
  sum_exp_scores = np.sum(exp_scores, 1)
  rlt_div = correct_exp_score / sum_exp_scores
  losses = -1.0 * np.log(rlt_div)
  loss = np.mean(losses)
  
  dloss = 1.0
  dlosses = np.ones((num_train, )) * dloss / num_train
  drlt_div = dlosses * -1.0 / rlt_div
  dcorrect_exp_score = drlt_div * 1.0/sum_exp_scores
  dsum_exp_scores = drlt_div * correct_exp_score * -1.0/ (sum_exp_scores ** 2)
  dexp_scores = np.ones((num_train, num_classes)) * dsum_exp_scores.reshape((num_train, 1))
  dexp_scores[range(num_train), y] += dcorrect_exp_score.reshape((num_train,))
  dscores = dexp_scores * np.exp(scores)

  dW += X.T.dot(dscores)
  dW += dloss * 0.5 * reg * 2 * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

