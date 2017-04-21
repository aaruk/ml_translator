import platform
import json
import pandas as pd
import ast
import numpy as np
import proto.neural_net as nn
import proto.word2vec as w2v
import proto.vec2word as v2w
import pickle as pkl

def create_one_hot_labels(labels, nn_labels):
  """
  Convert labels from integers to a "indicator" vector

  Eg: If label = 2, and possible classes = 3,
      function sets label_vec = [0,1,0]
  """

  # Currently hardcoding label dims
  lab_count = np.unique(labels).shape[0]
  print "Label Unique: ", lab_count
  indic_labels = np.zeros((labels.shape[0], lab_count))  # mx10 matrix
  for i in np.arange(labels.shape[0]):
    l = nn_labels[labels[i]]
    indic_labels[i, l] = indic_labels[i,l] = 1.
  return indic_labels

def create_cbow_tr_data(fpath="", win_size=1):
  """
  Reads text file and computes one hot vector for training
  """ 
  in_data = []
  # Read sentences line by line into list
  with open(fpath, 'r') as f:
    in_data = f.readlines()
  in_data = [x.strip("\n") for x in in_data]
  in_data = [x.strip("\xef\xbb\xbf") for x in in_data]

  vocab = " ".join(in_data)
  vocab = np.unique(vocab.split(" "))
  vocab_rows, vocab_cols = vocab.shape[0], vocab.shape[0]
  vocab_len = len(vocab)

  # Store indices of words from vocab in a dictionary 
  vocab_indices = {}
  for i in np.arange(vocab_rows):
    vocab_indices[vocab[i]] = i

  with open('datasets/europarl/eng_vocab_akns_150.json', 'wb') as mf:
    my_str = json.dumps(vocab_indices, encoding='utf-8')
    mf.writelines(my_str)
  mf.close()

  onehot_list = []
  cbow_data = []
  #win_size = 1 # window size for cooccurence matrix
  # Go through each word in vocabulary and update cooccurence matrix
  for i, sentence in enumerate(in_data):
    word_list = sentence.split(" ")
    for wi, word in enumerate(word_list):
      nbr_indices = []

      temp_list = []
      for ni in np.arange(wi-win_size, wi+win_size+1):
        nbr_indices.append(ni)
        temp_list.append(np.zeros((vocab_len), dtype=np.float32))
      
      ii = 0
      for ni in nbr_indices:
        if ni<0 or ni>=len(word_list):
          ii = ii+1
          continue
        else:
          word_ind = vocab_indices[word_list[ni]]
          temp_list[ii][word_ind] = 1
          ii = ii+1
        
      onehot_list.append(temp_list)
        
  for oi in np.arange(0,len(onehot_list)):
    temp1 = np.concatenate((onehot_list[oi][0],onehot_list[oi][2]))
    temp1 = np.concatenate((temp1,onehot_list[oi][1]))
    cbow_data.append(temp1)
  cbow_mat = np.array(cbow_data)
  m = 1
  return cbow_mat, vocab_len


def embed_words_cbow(tr_fpath="", retrain=True):
  """
  Function reads a mono-lingual corpus and learns word-embeddings using auto-encoder
  """
  train_mat, vocab_len = create_cbow_tr_data(tr_fpath, 3)
  mozhi_net = nn.NeuralNet(layer_count=3,
                           activation="linear",
                           out_activ="softmax",
                           max_epochs=5000,
                           eta=0.1)
  mozhi_net.init_layer(0, node_count=2*vocab_len, seq_node_cnt=0)
  mozhi_net.init_layer(1, node_count=180, seq_node_cnt=0)
  mozhi_net.init_layer(2, node_count=vocab_len, seq_node_cnt=0)
  mozhi_net.set_train_data(train_mat)
  # TODO: needed to print epoch-wise accuracy; Incomplete
  tr_labels = train_mat[:, -vocab_len:]
  tr_labels_orig = np.argmax(tr_labels, axis=1)
  mozhi_net.set_orig_labels (tr_labels_orig, tr_labels_orig)
  mozhi_net.fit()
  return
  
if __name__ == "__main__":
  if platform.system() == "Linux":
    usew2v = True
    
    #embed_words_cbow(tr_fpath="datasets/europarl/eng_150.txt",
    #                 retrain=True)
    # train_test_mozi_net_for_debugging()
    train_mat, vocab_len = create_cbow_tr_data("datasets/europarl/eng_150.txt", 3)
  else:
    embed_words_cbow(tr_fpath="datasets\\europarl\\eng_100.txt", 
                     retrain=True)
    # train_test_mozhi_net_for_debugging()
    # 
    
