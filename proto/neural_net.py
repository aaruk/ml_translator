"""
Module with a class for implementing neural network from scratch
"""
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.cm as cm

class NeuralNet(object):

  def __init__(self, layer_count=1,
                     activation="sigmoid",
                     max_epochs=100,
                     eta=0.1):
    """
    Class with attributes and functions required to train and test a neural
    network

    Inputs:
    ------
    layer_count : Total no of layers in neural net
                  Eg. layer_count=4 means 1 input layer, 1 output layer,
                                          2 hidden layers
    activation  : Type of activation to be used
                  Current options: sigmoid, tanh
    max_epochs  : Termination criterion-checked during training

    """
    self.layer_count = layer_count
    self.activation = activation
    self.epochs = max_epochs

    self.layer_wts = [None]*layer_count
    self.layer_dims = [0]*layer_count
    self.layer_grads = [None]*layer_count
    self.layer_dims = [0]*layer_count
    self.z          = [None]*layer_count
    self.eta        = eta

  def init_layer(self, layer_idx=None, layer_type=None, node_count=1,
                 layer_params=None):
    """
    Function initializes layers - setting layer types, layer_params,
    random weights. Has to be called as many times as there are layers

    Inputs:
    ------
    layer_idx   : index of layer to be intialized
    layer_type  : dense, convolutional, max_pool etc.
                  [Currently supports only dense layers are supported]
    node_count  : No of nodes in current layer
    layer_params: Parameters settings to be used in layers
    """
    if layer_idx == 0:
      # If idx=0, layer is layer_wts = input vector itself
      self.layer_wts[layer_idx] = np.array((node_count, 1))
      self.layer_dims[layer_idx] = node_count
    else:
      prev_node_count = self.layer_dims[layer_idx-1]+1 # Included Bias

      # wts are initialzed from a normal distribution
      self.layer_wts[layer_idx] = np.random.randn(node_count, prev_node_count)
      self.layer_grads[layer_idx] = np.zeros((node_count, prev_node_count), np.float32)

      # Hidden layer activation outputs
      self.z[layer_idx]           = np.zeros((node_count, 1), np.float32)
    self.layer_dims[layer_idx]    = node_count

    return

  def set_train_data(self, inp_data=None):
    """
    Function sets training data to be used for wt updation using
    backpropagation
    """
    self.train_data = inp_data
    self.train_targets = inp_data[:, -2:] #NOTE: Hardcoded - replace with output unit_count
    return

  def set_test_data(self, test_data=None):
    """
    Function sets training data to be used for wt updation using
    backpropagation
    """
    self.test_data = test_data
    self.test_targets = test_data[:, -2:] #NOTE: Hardcoded - replace with output unit_count
    return

  def sigmoid(self, x=None):
    """
    Sigmoidal activation function [-1,1]
    """
    ex = np.exp(-x)
    out = 1/(1+ex) 
    return out

  def predict(self, x):
    """
    Function predicts on a single training/testing sample using trained weights

    Do not call before calling NeuralNet.fit()
    """
    # Do forward prop across all layers in network
    # Handle special cases:
    # Layer 1 - Input layer
    # Layer n - Output layer (last layer)
    for idx in np.arange(self.layer_count):
      wts = self.layer_wts[idx]
      if idx == 0:
        self.z[idx] = x.reshape(x.shape[0], 1)
        continue
      else:
        xi = self.z[idx-1]
      xi = xi.reshape((xi.shape[0], 1))
      bias = np.ones((xi.shape[1], 1))
      xi = np.concatenate((bias, xi), axis=0)
      wx = np.matmul(wts, xi)
      # Handle different activation types cleanly
      #NOTE: Make sure to do appropriate changes to backprop
      if self.activation=="sigmoid":
        self.z[idx] = self.sigmoid(wx)

    out = self.z[self.layer_count-1]
    return out

  def predict_all(self, inp_array=None):
    """
    Function predicts on an array of samples

    Input : [mxn] Array of samples m-no of samples; n-dim of input data
    ------
    Output: [mxn] Array of predictions m-no of samples; n-dim of output data
    -------
    """
    return pred_mat

  def backprop(self, target):
    """
    Mostly for internal usage. Function backpropagates weights across layers
    """
    # Start from last layer and compute gradients across layers
    # i-index of layer
    dw_list = [] # Hold list of wt gradients to be deducted from each layer wts

    # Backpropagate weights, starting from last layer
    # Note that no of wt matrices is 1 less than layer count
    for i in np.arange(self.layer_count-1, 0, step=-1):
      z = self.z[i]
      z_prev = self.z[i-1]
      bias = np.ones((z_prev.shape[1], 1))
      z_prev = np.concatenate((bias, z_prev), axis=0)
      if i == (self.layer_count-1):
        # TODO: Update here if activation function is changed
        if self.activation == "sigmoid":
          self.layer_grads[i] = (target-z)*z*(1-z)
      else:
        # Note that wt from next layer has to be used (before wt updation)
        wt_next = self.layer_wts[i+1][:, 1:]
        grad_next = self.layer_grads[i+1]
        self.layer_grads[i] = np.matmul(wt_next.T, grad_next)*z*(1-z)
      dw = np.matmul(self.layer_grads[i], z_prev.T) 
      dw_list.append(dw)

    # Update all weights together to avoid confusion
    # self.layer_wts[1:] = self.layer_wts[1:]-dw_list
    for l in np.arange(self.layer_count-1,  0, step=-1):
      self.layer_wts[i] =  self.layer_wts[i]-self.eta*dw_list[i+1]
    return

  def fit(self):
    """
    Function trains neural network using training data & network params
    set previously. Does a stochastic gradient descent on training samples.
    """
    tr_pred_list = []
    for e in np.arange(self.epochs):
      for idx, x in enumerate(self.train_data):
        self.out = self.predict(x)
        tr_pred_list.append(self.out)
        target = self.train_targets[idx]
        target = target.reshape(target.shape[0], 1)
        self.backprop(target)
    return
# End of class Neural Network
