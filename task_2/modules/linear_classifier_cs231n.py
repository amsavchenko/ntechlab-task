import torch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3, 
                            reg=1e-5, num_iters=100, batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_classes = torch.max(y) + 1
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        y_batch = None
        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in this round of gradient descent.        #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
        # and y_batch should have shape (batch_size,)                           #
        #                                                                       #
        # Hint: Use torch.randint to generate indices.                          #
        #########################################################################
        # Replace "pass" statement with your code
        batch_indices = torch.randint(low=0, high=num_train, size=(batch_size, ))
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # Replace "pass" statement with your code
        W -= learning_rate * grad
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Replace "pass" statement with your code
    y_pred = X.matmul(W).max(axis=1).indices
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


class LinearClassifier(object):
  
    def __init__(self):
        self.W = None
    
    def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                      num_iters, batch_size, verbose)
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X):
        return predict_linear_classifier(self.W, X) 

    def loss(self, W, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        pass
    
    def _loss(self, X_batch, y_batch, reg):
        self.loss(self.W, X_batch, y_batch, reg)


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.  When you implment the 
    regularization over W, please DO NOT multiply the regularization term by 1/2 
    (no coefficient). 

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    scores = X.matmul(W)
    # fix numeric instability
    scores -= scores.max(axis=1).values.view(-1, 1)
    probs = scores.exp() / scores.exp().sum(axis=1).view(-1, 1)
    correct_class_probs = probs[range(num_train), y]
    loss = (correct_class_probs.log() * (-1)).sum()

    loss /= num_train
    loss += reg * (W * W).sum()

    probs[range(num_train), y] -= 1
    dW = X.T.matmul(probs)

    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """
    def loss(self, W, X_batch, y_batch, reg):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)
