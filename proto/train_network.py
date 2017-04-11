import numpy as np
import proto.neural_net as nn
import proto.word2vec as w2v
import proto.vec2word as v2w
import pickle as pkl

def train_test_mozhi_net_old(train_data, test_data):
  """
  Train and test a given neural net architecture
  """
  # Initialize network architecturerun
  mozhi_net = nn.NeuralNet(4, max_epochs=2)
  mozhi_net.init_layer(0, node_count=2)
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
  
def train_test_mozhi_net(tr_fpath=None, tst_fpath=None, retrain=True, 
                         reenc=False,
                         wvecs_len=6,
                         wvecs_en_fpath=None,
                         vocab_en_fpath=None,
                         cc_mat_en_fpath=None,
                         wvecs_fr_fpath=None,
                         vocab_fr_fpath=None,
                         cc_mat_fr_fpath=None):
  """
  Train and test a given neural net architecture
  
  Input:
  ------
  tr_fpath       : training data file path, containing sentences. This should be 
                   the folder containing both the english and french text files.
  tst_fpath      : testing data file path, conatining sentences
  retrain        : True if you want the neural network to be retrained
  reenc          : True if you want the words to be rencoded into vectors
  wvecs_len      : This will decide the dimension of the vector representation
  wvecs_en_fpath : File path to be used to store the vector rep for english
  vocab_en_fpath : File path to be used to store the vocabulary for english
  cc_mat_en_fpath: File path to be used to store co occurrence matrix for eng 
  wvecs_fr_fpath : File path to be used to store the vector rep for french
  vocab_fr_fpath : File path to be used to store the vocabulary for french
  cc_mat_fr_fpath: File path tone used to store co occurrence matrix for french
  
  """
  # Initialize network architecture
  mozhi_net = nn.NeuralNet(layer_count=4, max_epochs=1000,eta=0.7)
  mozhi_net.init_layer(0, node_count=wvecs_len)
  mozhi_net.init_layer(1, node_count=40)
  mozhi_net.init_layer(2, node_count=40)
  mozhi_net.init_layer(3, node_count=wvecs_len)
  
  
  if retrain:
    # Retrain the neural network with the training data received.
    if reenc:
      # Reencode the words into their feature vector representation.
      tr_wvecs_en, tr_vocab_en, tr_cc_mat_en = w2v.word2vec(tr_fpath+"eng.txt")
      tr_wvecs_fr, tr_vocab_fr, tr_cc_mat_fr = w2v.word2vec(tr_fpath+"frn.txt")
      np.save(wvecs_en_fpath,tr_wvecs_en) # Saving the entire umatrix
      np.save(vocab_en_fpath,tr_vocab_en)
      np.save(cc_mat_en_fpath,tr_cc_mat_en)
      np.save(wvecs_fr_fpath,tr_wvecs_fr)
      np.save(vocab_en_fpath,tr_vocab_fr)
      np.save(cc_mat_fr_fpath,tr_cc_mat_fr)
      # Choose top wvecs_len singular vectors
      tr_wvecs_en = tr_wvecs_en[:,:wvecs_len]
      tr_wvecs_fr = tr_wvecs_fr[:,:wvecs_len]
      
    else:
      # load feature vectors from encoding done previously
      # this part of the code for else needs to be checked.
      tr_wvecs_en = np.load(wvecs_en_fpath)
      tr_vocab_en = np.load(vocab_en_fpath)
      tr_cc_mat_en = np.load(cc_mat_en_fpath)
      tr_wvecs_fr = np.load(wvecs_fr_fpath)
      tr_vocab_fr = np.load(vocab_en_fpath)
      tr_cc_mat_fr = np.load(cc_mat_fr_fpath)
      #Choose top wvecs_len singular vectors
      tr_wvecs_en = tr_wvecs_en[:,:wvecs_len]
      tr_wvecs_fr = tr_wvecs_fr[:,:wvecs_len]
    
    with open(tr_fpath+"eng.txt", 'r') as f:
      tr_data_en = f.readlines()
    tr_data_en = [x.strip("\n") for x in tr_data_en]
    # Removing some additional characters it is adding to the file.
    tr_data_en = [x.strip("\xef\xbb\xbf") for x in tr_data_en]
    print tr_data_en
    tr_data_en = " ".join(tr_data_en)
    # List which contains all the words from the training sentences
    tr_data_en = tr_data_en.split(" ") 
    
    with open(tr_fpath+"frn.txt", 'r') as f:
      tr_data_fr = f.readlines()
    tr_data_fr = [x.strip("\n") for x in tr_data_fr]
    # Removing some additional characters it is adding to the file.
    tr_data_fr = [x.strip("\xef\xbb\xbf") for x in tr_data_fr]
    tr_data_fr = " ".join(tr_data_fr)
    tr_data_fr = tr_data_fr.split(" ")

    # This will be the matrix containing the vector representation of both
    # english and french. With first wvec_len columns being english and the
    # rest french.
    tr_data = np.zeros((len(tr_data_en),2*wvecs_len), dtype=np.float32)
    
    for i in np.arange(0, len(tr_data_en)):
      # Assuming that all the words in the training set can be found in vocab
      tr_data[i,:wvecs_len] = tr_wvecs_en[tr_vocab_en[tr_data_en[i]],:]
    
    for i in np.arange(0, len(tr_data_fr)):
      # Assuming that all the words in the training set can be found in vocab
      tr_data[i,wvecs_len:] = tr_wvecs_fr[tr_vocab_fr[tr_data_fr[i]],:]
      
    # Set training and testing data to be used
    mozhi_net.set_train_data(tr_data)
    # Searches for optimal weights using stochastic gradient descent
    mozhi_net.fit()
    pkl.dump(mozhi_net,open("mozhi_net_file.pkl","wb"))
    
  else:
    print "To be done"
    # Here we need to retreive all the data from pkl and assign it to the object
    # so that it can be used for testing (like weights).
  
  # Creating a list of lists for the test data (sentence list with words as a 
  # list inside). 
  tst_data_list = []
  # This will have the equivalent vector representation of every word in english
  tst_data = []
  # This will contain vector representation of every word in french. 
  tst_target = []
  # This list will contain the equivalent french sentences.
  tst_target_fr = []
  
  # These 2 temporary variables help in storing the vector representation before
  # appending to the main list.
  temp1_list = []
  temp1 = np.zeros((1,wvecs_len),np.float32)
  # These 2 are used only for prediction
  temp2_list = []
  temp2 = np.zeros((1,wvecs_len),np.float32)
  
  # Creating list of list with sentences and words.
  with open(tst_fpath+"eng.txt", 'r') as f:
    tst_data_en = f.readlines()
  tst_data_en = [x.strip("\n") for x in tst_data_en]
  # Removing some additional characters it is adding to the file.
  tst_data_en = [x.strip("\xef\xbb\xbf") for x in tst_data_en]
  for i in np.arange(0, len(tst_data_en)):
    tst_data_list.append(tst_data_en[i].split(" "))
    
  # Creating the equivalent list of list with vector representations.
  for i in np.arange(0, len(tst_data_list)):
    for j in np.arange(0, len(tst_data_list[i])):
      # Assuming that all the words in the testing set can be found.
      temp1 = tr_wvecs_en[tr_vocab_en[tst_data_list[i][j]],:]
      temp1_list.append(temp1)
    tst_data.append(temp1_list)
    temp1_list = [] # Clearing the list for next iteration.  
  
  temp1_list = []
  
  # Predicting
  for i in np.arange(0,len(tst_data)):
    for j in np.arange(0,len(tst_data[i])):
      temp2 = mozhi_net.predict(tst_data[i][j])
      # Transpose as predict returns column vector.
      temp2_list.append(temp2.T)
      print temp2.T
    tst_target.append(temp2_list)
    temp2_list = []
  
  
  # Decoder
  for i in np.arange(0,len(tst_target)):
    for j in np.arange(0,len(tst_target[i])):
      temp1 = v2w.vec2word(in_vec = tst_target[i][j], vocab = tr_vocab_fr,
                           word2vec_arr = tr_wvecs_fr)
      temp1_list.append(temp1)
    tst_target_fr.append(temp1_list)
    temp1_list = []
    
  print tst_target_fr
  
  return

 


if __name__ == "__main__":
  train_test_mozhi_net(tr_fpath="data\\akns\\set_01\\", 
                       tst_fpath="data\\akns\\set_01\\",
                       retrain=True, 
                       reenc=True,
                       wvecs_len=8,
                       wvecs_en_fpath="data\\wvecs_en_file.npy",
                       vocab_en_fpath="data\\vocab_en_file.npy",
                       cc_mat_en_fpath="data\\cc_mat_en_file.npy",
                       wvecs_fr_fpath="data\\wvecs_fr_file.npy",
                       vocab_fr_fpath="data\\vocab_fr_file.npy",
                       cc_mat_fr_fpath="data\\cc_mat_fr_file.npy")
