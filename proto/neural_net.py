"""
Module with a class for implementing neural network from scratch
"""
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.cm as cm

class NeuralNet(object):

  def __init__(self, layer_count=1,
                     activation="sigmoid",
                     out_activ="softmax",
                     max_epochs=100,
                     eta=0.1,
                     phr_wrd_cnt=1):
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
    phr_wrd_cnt  : number of words in the phrase when splitting the sentence

    """
    self.layer_count = layer_count
    self.activation = activation
    self.out_activation = out_activ
    self.epochs = max_epochs
    self.phr_wrd_cnt = phr_wrd_cnt

    self.layer_wts = [None]*layer_count
    self.layer_dims = [0]*layer_count
    self.layer_grads = [None]*layer_count
    self.layer_dims = [0]*layer_count
    self.z          = [None]*layer_count
    self.eta        = eta
    self.seq_node_cnt_list = [0]*layer_count
    self.train_labels_orig = None
    self.test_labels_orig = None


  def init_layer(self, layer_idx=None, layer_type=None, node_count=1,
                 layer_params=None,
                 seq_node_cnt=0):
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
    seq_node_cnt: wordvec length, this will help in creating that many extra 
                  nodes for the input.
    """
    if layer_idx == 0:
      # If idx=0, layer is layer_wts = input vector itself
      self.layer_wts[layer_idx] = np.array((node_count, 1))
      self.layer_dims[layer_idx] = node_count
      # We don't want to add any additional nodes to input layer
      self.seq_node_cnt_list[layer_idx] = 0
    else:
      self.seq_node_cnt_list[layer_idx] = seq_node_cnt
      # Included Bias and input word
      prev_node_count = self.layer_dims[layer_idx-1]+self.seq_node_cnt_list[layer_idx-1]+1

      # wts are initialzed from a normal distribution
      # self.layer_wts[layer_idx] = np.random.rand(node_count, prev_node_count)
      self.layer_wts[layer_idx] = np.random.uniform(low=-0.5, high=0.5,\
                                                    size=(node_count, prev_node_count))
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
    inp_dim = self.layer_dims[0]
    out_dim = self.layer_dims[self.layer_count-1]
    self.train_data = inp_data[:, :inp_dim]
    self.train_targets = inp_data[:, inp_dim:]
    return


  def set_test_data(self, test_data=None):
    """
    Function sets training data to be used for wt updation using
    backpropagation
    """
    inp_dim = self.layer_dims[0]
    out_dim = self.layer_dims[self.layer_count-1]
    self.test_data = test_data[:, :inp_dim]
    self.test_targets = test_data[:, inp_dim:]
    return


  def set_orig_labels(self, tr_labels_orig, tst_labels_orig):
    """
    Function sets integer labels [0,1,2,3...] instead of the "Indicator" labels
    used for neural  net training"
    """

    self.train_labels_orig = tr_labels_orig
    self.test_labels_orig  = tst_labels_orig
    return


  def sigmoid(self, x=None):
    """
    Sigmoidal activation function [0,1]
    """
    ex = np.exp(-x)
    out = 1/(1+ex) 
    return out


  def act_tanh(self, x=None):
    """
    Tanh activation function [-1,1]
    """
    ex = np.exp(-x)
    out = (1-ex)/(1+ex) 
    return out


  def softmax(self, x=None):
    """
    Softmax activation function
    
    Important NOTE: Softmax applied to entire layer at once
    """
    ex  = abs(np.exp(x))
    denom = ex.sum()
    if denom > (10**(-4)):
      out_vec = (1/denom)*(ex)
    else:
      out_vec = np.zeros(x.shape) 
    #print out_vec.T, 
    return out_vec


  def predict(self, x):
    """
    Function predicts on a single training/testing sample using trained weights

    Do not call before calling NeuralNet.fit()
    """
    # Do forward prop across all layers in network
    # Handle special cases:
    # Layer 1 - Input layer
    # Layer n - Output layer (last layer)
    # The input x will be of dimension phr_wrd_cnt X wrdveclen
    # The ouput will also be of the same dimension
    kk = 1 #this index is used to concatenate the new word with that hidden layer
    for idx in np.arange(self.layer_count):
      wts = self.layer_wts[idx]
      if idx == 0:
        self.z[idx] = x[0,:].reshape(x.shape[1], 1)
        continue
      else:
        xi = self.z[idx-1]
      xi = xi.reshape((xi.shape[0], 1))
      bias = np.zeros((xi.shape[1], 1))
      if self.seq_node_cnt_list[idx-1]>0:
        xi = np.concatenate((bias,x[kk,:].reshape(x.shape[1], 1), xi), axis=0)
        kk = kk+1
      else:
        xi = np.concatenate((bias, xi), axis=0)
      wx = np.matmul(wts, xi)
      #print "wx: "
      #print wx.T
      # Handle different activation types cleanly
      #NOTE: Make sure to do appropriate changes to backprop
      if self.activation=="sigmoid":
        self.z[idx] = self.sigmoid(wx)
      elif self.activation=="tanh":
        self.z[idx] = self.act_tanh(wx)
      elif self.activation=="linear":
        self.z[idx] = wx/2.0

      # Apply softmax only if it is last layer
      if idx == (self.layer_count-1):
        if self.out_activation == "softmax":
          self.z[idx] = self.softmax(wx)
        
    #out = np.zeros((x.shape[0],x.shape[1]))
    out = np.zeros((1, self.layer_dims[self.layer_count-1]))
    nn = 0 #index for tracking row of out matrix to append
    for mm in np.arange(self.layer_count-self.phr_wrd_cnt,self.layer_count):
      out[nn,:] = self.z[mm].reshape(1,self.z[mm].shape[0])
      nn = nn+1
    return out


  def predict_all(self, inp_data=None):
    """
    Function predicts on an array of samples

    Input : [mxn] Array of samples m-no of samples; n-dim of input data
    ------
    Output: [mxn] Array of predictions m-no of samples; n-dim of output data
    -------
    """

    # Test trained model
    preds = []
    for i in np.arange(inp_data.shape[0]):
      x = np.reshape(inp_data[i], (1, inp_data[i].shape[0]))
      nn_pred = self.predict(x)
      dig_pred = np.argmax(nn_pred)
      #print nn_pred, " ---> "
      #print dig_pred, "  "
      preds.append(dig_pred)
    preds = np.array(preds)
    return preds


  def backprop(self, target_mat, source_mat):
    """
    Mostly for internal usage. Function backpropagates weights across layers
    """
    # Start from last layer and compute gradients across layers
    # i-index of layer
    dw_list = [] # Hold list of wt gradients to be deducted from each layer wts

    # Backpropagate weights, starting from last layer
    # Note that no of wt matrices is 1 less than layer count
    for kk in np.arange(0,self.phr_wrd_cnt):
      # This for loop is to propogate the error from different output layers in 
      # the sequence to sequence case
      target = target_mat[kk,:].reshape(target_mat.shape[1],1)
      for i in np.arange(self.layer_count-1-kk, 0, step=-1):
        z = self.z[i]
        z_prev = self.z[i-1]
        bias = np.zeros((z_prev.shape[1], 1))

        if self.seq_node_cnt_list[i-1]>0:
          z_prev = np.concatenate((bias, source_mat[i-1,:].reshape(source_mat.shape[1],1), z_prev), axis=0)
        else:
          z_prev = np.concatenate((bias, z_prev), axis=0)

        if i == (self.layer_count-1):
          # TODO: Update here if activation function is changed
          if self.out_activation == "sigmoid":
            self.layer_grads[i] = (target-z)*z*(1-z)
          elif self.out_activation == "tanh":
            self.layer_grads[i] = (target-z)*(1-z)*(1+z)*(1/2.0)
          if self.out_activation == "softmax":
            self.layer_grads[i] = (target-z)
            #print z.T, target.T
        else:
          # Note that wt from next layer has to be used (before wt updation)
          if self.seq_node_cnt_list[i]>0:
            wt_next = self.layer_wts[i+1][:, 1+self.seq_node_cnt_list[i]:]
          else:
            wt_next = self.layer_wts[i+1][:,1:]
          grad_next = self.layer_grads[i+1]
          if self.activation == "sigmoid":
            self.layer_grads[i] = np.matmul(wt_next.T, grad_next)*z*(1-z)
          elif self.activation == "tanh":
            self.layer_grads[i] = np.matmul(wt_next.T, grad_next)*(1-z)*(1+z)*(1/2.0)
          elif self.activation == "linear":
            self.layer_grads[i] = np.matmul(wt_next.T, grad_next)/2.0
        dw = np.matmul(self.layer_grads[i], z_prev.T) 
        dw_list.append(dw)

    # Update all weights together to avoid confusion
    # self.layer_wts[1:] = self.layer_wts[1:]-dw_list
    # dw_list say we had 7 layers, and a phrase count of 3, we will have the 
    # following list dw_6,dw_5,dw_4,dw_3,dw_2,dw_1,dw_5,...,dw_1,dw_4,...,dw_1
    jj = 0
    for kk in np.arange(0,self.phr_wrd_cnt):
      for l in np.arange(self.layer_count-1-kk,  0, step=-1):
        self.layer_wts[l] = self.layer_wts[l]+self.eta*dw_list[jj]
        #self.layer_wts[l][:, 1:] = self.layer_wts[l][:, 1:] - 0.0001*self.layer_wts[l][:, 1:]
        jj += 1
    

    return

  def fit(self):
    """
    Function trains neural network using training data & network params
    set previously. Does a stochastic gradient descent on training samples.
    """
    tr_pred_list = []
    x = np.zeros((self.phr_wrd_cnt,self.train_data.shape[1]))
    for e in np.arange(self.epochs):
      for idx in np.arange(0,self.train_data.shape[0]/self.phr_wrd_cnt):
        ii = idx*self.phr_wrd_cnt
        x = self.train_data[ii:ii+self.phr_wrd_cnt,:]
        self.out = self.predict(x)
        tr_pred_list.append(self.out)
        target = self.train_targets[ii:ii+self.phr_wrd_cnt,:]
        #print target
        self.backprop(target,x)

      if self.eta >= 0.0001:
        if e%50 == 0:
          self.eta = self.eta / 10.

      # Compute training & testing accuracy once every epoch
      tr_preds = self.predict_all(self.train_data)
      tr_labels_orig = self.train_labels_orig
      tr_acc = np.array((tr_preds == tr_labels_orig), np.uint8)
      total_tr_acc = tr_acc.sum()
      tr_acc = float(total_tr_acc) / tr_labels_orig.shape[0] * 100

      #tst_labels_orig = self.test_labels_orig
      #tst_preds = self.predict_all(self.test_data)
      #tst_acc = np.array((tst_preds == tst_labels_orig), np.uint8)
      #total_tst_acc = tst_acc.sum()
      #tst_acc = float(total_tst_acc) / tst_labels_orig.shape[0] * 100
      print "Epoch : ", e, " | Training Accuracy: ", tr_acc, "%"
      np.savetxt("results/tr_labels_orig_"+str(e)+".txt", tr_labels_orig, fmt="%.0f")
      np.savetxt("results/tr_preds_"+str(e)+".txt", tr_preds, fmt="%.0f")

      # Save network wts for each epoch
      for i in np.arange(self.layer_count):
        np.savetxt("models/layer_wts_"+str(i)+"_e"+str(e)+".txt", self.layer_wts[i], delimiter=", ")

    return
# End of class Neural Network
