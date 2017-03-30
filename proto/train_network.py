import numpy as np
import proto.neural_net as nn

def train_test_mozhi_net(train_data, test_data):
  """
  Train and test a given neural net architecture
  """
  # Initialize network architecture
  mozhi_net = nn.NeuralNet(4, max_epochs=2)
  mozhi_net.init_layer(0, node_count=2)
  mozhi_net.init_layer(1, node_count=4)
  mozhi_net.init_layer(2, node_count=3)
  mozhi_net.init_layer(3, node_count=2)

  # Set traiing and testing data to be used
  mozhi_net.set_train_data(train_data)
  mozhi_net.set_test_data(test_data)

  # Searches for optimal weights using stochastic gradient descent
  mozhi_net.fit()
  #train_preds = mozhi_net.get_train_preds()
  #test_preds  = mozhi_net.get_test_preds()

  return


if __name__ == "__main__":
  train_data = np.load("data/word_vec.npy")
  train_data = train_data[:, :2]
  test_data = np.load("data/word_vec.npy")
  test_data = test_data[:, :2]
  train_test_mozhi_net(train_data, test_data)
