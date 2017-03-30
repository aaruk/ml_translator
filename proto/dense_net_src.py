"""
Module with functions for forward and backward propagation
"""
import numpy as np


def sigmoid(x=None):
  """
  Sigmoidal activation function [-1,1]
  """
  ex = np.exp(-x)
  out = (1-ex)/(1+ex) 
  return out


def predict(x, wt_h1, wt_o):
  hidden_node_count = 10
  xw1 = np.matmul(wt_h1, x)  # 10x1
  h1 = sigmoid(xw1) # 10x1
  h1 = np.append([[1]], h1)
  h1 = h1.reshape(hidden_node_count+1, 1)  # 11x1
  h2 = np.matmul(wt_o.T, h1) # 1x1
  y = sigmoid(h2)  # 1x1
  if (y[0,0] < 0):
	  return 1
  else:
	  return -1

  return 0


def train(train_samples, epochs=10):
  """
  Function to train neural network with single hidden layer
  """
  #wt_h1 = np.random.uniform(0, 1, (hidden_node_count, 3))
  #wt_o  = np.random.uniform(0, 1, (hidden_node_count+1, 1))
  #wt_h1 = np.random.randn(0, 1, (hidden_node_count, 3))
  wt_o  = np.random.randn(11,1)
  wt_h1  = np.random.randn(10,3)

  train_sample_count = train_samples.shape[0]
  eta = 0.6 # Learning Rate

  np.savetxt('results/train_dat.csv', train_samples, fmt='%.01f', delimiter=', ')
  test_data = gen_data(0, 1, 100)
  np.savetxt('results/test_dat.csv',  test_data, fmt='%.01f', delimiter=', ')
  err = [] 
  hidden_node_count = 10
  plot_step = 0.007
  reg = 0.0000 # Regularization param
  for i in np.arange(epochs):
    # Compute Activations layer by layer
    # Update wts using stochastic gradient descent
    for j in np.arange(train_sample_count):
      x = train_samples[j, :-1].reshape(3, 1)  # 3x1
      xw1 = np.matmul(wt_h1, x)  # 10x1
      h1 = sigmoid(xw1) # 10x1
      h1 = np.append([[1]], h1)
      h1 = h1.reshape(hidden_node_count+1, 1)  # 11x1
      h2 = np.matmul(wt_o.T, h1) # 1x1
      y = sigmoid(h2)  # 1x1
      #if y[0, 0] > 0.5:
      #  y[0, 0] = 1
      #else:
      #  y[0, 0] = -1

      # Backprop: update wts corresponding to last layer
      t = train_samples[j,-1]  # 1x1
      dwt_o = (t-y)*(1+y)*(1-y)*h1/2.0  # 11x1
      tmp_wt_o = wt_o.copy()  # 11x1
      # tmp_wt_o = tmp_wt_o[1:, 0] # 10x1  #NOTE: Might need to do this
      wt_o = wt_o + eta*dwt_o # 10x1
      wt_o[1:] = wt_o[1:] - reg*tmp_wt_o[1:]

      # Backprop: update wts corresponding to penultimate layer
      h1 = h1[1:]
      del1 = (t-y)*(1+y)*(1-y)*wt_o[1:]*(1+h1)*(1-h1)/4.0 # 10x1
      dwt_h1 = np.matmul(del1, x.T) # [10x1]*[1x3] = [10x3]
      wt_h1_prev = wt_h1.copy()
      wt_h1 = wt_h1 + eta*dwt_h1 # [10x3]
      wt_h1[:, 1:3] = wt_h1[:, 1:3] - reg*wt_h1_prev[:, 1:3]
      # print wt_h1

      # Update network weights layer by layer
      for  l in layer_count:
        if l == layer_count-1:
          dw[l] = (t[i]-z[l][i])*z[l]*(1-z[l])
        else:
          dw_cur = wt[l-1].T*z[l]*(1-z[l])
          dw[l] = dw[l-1] * dw_cur
        dw_upd = dw[l]

    # end for

    # Plot decision boundaries learnt by neurons
    plt.ioff()
    dbnd = plt.figure()
    dbnd.hold(True)
    for ind in np.arange(wt_h1.shape[0]):
      px1 = np.arange(0.1, 1, 0.1)
      px2 = - (px1*(wt_h1[ind,1]) + wt_h1[ind,0]) / wt_h1[ind,2]
      plt.plot(px1, px2)
    plt.title("Decision Boundaries: "+str(i))
    plt.xlabel("x1")
    plt.xlabel("x2")
    dbnd.savefig("results/dbnd_"+str(i)+".png")
    plt.close()
      
    # Yet another viusalizaition of decision learnt boundary 
    x_min, x_max = 0, 1 # X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = 0, 1 # X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = np.zeros(xx.shape)
    for ii in np.arange(xx.shape[0]):
      for jj in np.arange(xx.shape[1]):
        xq = np.array([1, xx[ii, jj], yy[ii, jj]]).reshape(3, 1)
        Z[ii,jj] = predict(xq, wt_h1, wt_o)
    #Z = predict(wt_h1, wt_o, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Learnt Decision Boundary")
    #plt.scatter(test_data[:,1], test_data[:,2])
    plt.savefig("results/decision_"+str(i)+".png")

    #print wt_h1[:, 1]
    err.append(test(test_data, wt_h1, wt_o))
    # end for

  np.savetxt("results/wth.mat", wt_h1)
  np.savetxt("results/wt_o.mat", wt_o)
  err = np.array(err)
  plt.ioff()
  plt.plot(err)
  plt.xlabel("epoch")
  plt.ylabel("error")
  plt.title("Test Error Plot")
  plt.savefig("results/test_error.png")
  np.savetxt("results/err.mat",err)
  return wt_h1, wt_o


def test(test_data, wt_h1, wt_o):
  """
  Test network using randomly generated data
  """
  y_vec = []
  error = 0
  test_sample_count = test_data.shape[0]
  hidden_node_count = wt_h1.shape[0]
  for j in np.arange(test_sample_count):
    x = test_data[j, :-1].reshape(3, 1)  # 3x1
    xw1 = np.matmul(wt_h1, x)  # 10x1
    h1 = sigmoid(xw1) # 10x1
    h1 = np.append([[1]], h1)
    h1 = h1.reshape(hidden_node_count+1, 1)  # 11x1
    h2 = np.matmul(wt_o.T, h1) # 1x1
    y = sigmoid(h2)  # 1x1
    #y_vec.append(y[0,0])
    y_vec.append(y)
    if y[0, 0] > 0.0:
      y[0, 0] = 1
    else:
      y[0, 0] = -1
    if (y[0,0] != test_data[j, -1]):
      #print y, test_data[j,-1]
      error += 1 

  #t_vec = test_data[:,3]
  y_vec = np.array(y_vec).reshape(test_sample_count, 1)
  res = np.concatenate((test_data, y_vec), axis=1)
  np.savetxt('results/res.csv', res, fmt='%.01f', delimiter=', ')


  #print "Done with this round"
  #print y_vec
  return error


