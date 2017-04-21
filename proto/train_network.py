import platform
import numpy as np
import proto.neural_net as nn
import proto.word2vec as w2v
import proto.vec2word as v2w
import pickle as pkl

def text2mat(text_fpath=None, wvecs_en=None, vocab_en=None, wvecs_fr=None, 
             vocab_fr=None,
             phr_wrd_cnt=1,
             wvecs_len=8):
  """
  Converts english and french sentences into a matrix with equivalent feature 
  vectors and padding.
  Input:
  ------
  text_fpath    : This should be the folder containing both the english and 
                  french sentence text files.
  wvecs_en      : Feature vectors for English
  vocab_en      : Dictionary containing word indices for English
  wvecs_fr      : Feature vectors for French
  vocab_fr      : Dictionary containing word indices for French
  phr_wrd_cnt    : Number of words in the phrase when breaking the sentence
  wvecs_len      : This will decide the dimension of the vector representation
  """
  sent_en_list = [] #List of list which will contain english sentences
  sent_fr_list = [] #List of list which will contain french sentences
  
  # Creating a list of list for English
  with open(text_fpath+"eng.txt", 'r') as f:
    data_en = f.readlines()
  data_en = [x.strip("\n") for x in data_en]
  # Removing some additional characters it is adding to the file.
  data_en = [x.strip("\xef\xbb\xbf") for x in data_en]
  for i in np.arange(0, len(data_en)):
   sent_en_list.append(data_en[i].split(" "))
  
  # Creating a list of list for French
  with open(text_fpath+"frn.txt", 'r') as f:
    data_fr = f.readlines()
  data_fr = [x.strip("\n") for x in data_fr]
  # Removing some additional characters it is adding to the file.
  data_fr = [x.strip("\xef\xbb\xbf") for x in data_fr]
  for i in np.arange(0, len(data_fr)):
   sent_fr_list.append(data_fr[i].split(" "))
   
  # Creating a list with padding included.
  wvecs_en_list = [] #This will be a list containing the feature vectors with padding
  wvecs_fr_list = [] #This will be a list containing the feature vectors with padding
  
  #List which contains the length of sentences and length with padding
  sent_len_en_list = []
  sent_len_fr_list = []
  
  padding = np.zeros(wvecs_len, dtype=np.float32)
  
  for s_idx in np.arange(0,len(sent_en_list)):
    # Since both english and french will have the same number of sentences, I 
    # I am using just the english list length for the loop.
    sent_len_en = len(sent_en_list[s_idx]) # Sentence length(no. of words)
    sent_len_fr = len(sent_fr_list[s_idx]) # Sentence length(no. of words)
    phr_num_en = float(sent_len_en)/phr_wrd_cnt # Number of phrases the sentence can be broken into
    phr_num_en = np.ceil(phr_num_en)
    phr_num_fr = float(sent_len_fr)/phr_wrd_cnt
    phr_num_fr = np.ceil(phr_num_fr)
    sent_len_en_list.append((sent_len_en,int(phr_num_en*phr_wrd_cnt)))
    sent_len_fr_list.append((sent_len_fr,int(phr_num_fr*phr_wrd_cnt)))
    if phr_num_en>=phr_num_fr:
      phr_num_fr = phr_num_en
    else:
      phr_num_en = phr_num_fr
    
    for w_idx in np.arange(0,int(phr_num_en*phr_wrd_cnt)):
      # Could have used as well phr_num_fr as both should be same with max value assigned
      if w_idx<sent_len_en:
        temp3 = wvecs_en[vocab_en[sent_en_list[s_idx][w_idx]],:]
      else:
        temp3 = padding
      wvecs_en_list.append(temp3)
      
      if w_idx<sent_len_fr:
        temp4 = wvecs_fr[vocab_fr[sent_fr_list[s_idx][w_idx]],:]
      else:
        temp4 = padding
      wvecs_fr_list.append(temp4)
      
  wvecs_en_mat = np.array(wvecs_en_list) #Wordvec matrix for english with padding
  wvecs_fr_mat = np.array(wvecs_fr_list) #Wordvec matrix for french with padding
  
  # This will be the matrix containing the vector representation of both
  # english and french. With first wvecs_len columns being english and the
  # rest french. This will also include the padding inbetween as needed.
  data_mat = np.zeros((len(wvecs_en_list),2*wvecs_len), dtype=np.float32)
  data_mat = np.concatenate((wvecs_en_mat, wvecs_fr_mat), axis=1)
  
  return data_mat, sent_len_en_list, sent_len_fr_list
  
def train_test_mozhi_net(tr_fpath=None, tst_fpath=None, retrain=True, 
                         reenc=False,
                         wvecs_len=8,
                         phr_wrd_cnt=1,
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
  phr_wrd_cnt    : Number of words in the phrase when breaking the sentence
  wvecs_en_fpath : File path to be used to store the vector rep for english
  vocab_en_fpath : File path to be used to store the vocabulary for english
  cc_mat_en_fpath: File path to be used to store co occurrence matrix for eng 
  wvecs_fr_fpath : File path to be used to store the vector rep for french
  vocab_fr_fpath : File path to be used to store the vocabulary for french
  cc_mat_fr_fpath: File path tone used to store co occurrence matrix for french
  
  """
  # Initialize network architecture
  # say the phr_wrd_cnt=2, this means you will have 2 words as input. One going 
  # to inout layer and the second one going in parallel to the first hidden layer.
  # So seq_node_cnt=wvecs_len for layer1.
  mozhi_net = nn.NeuralNet(7, activation="tanh", max_epochs=5000, eta=0.01, phr_wrd_cnt=phr_wrd_cnt)
  mozhi_net.init_layer(0, node_count=wvecs_len, seq_node_cnt=0)
  mozhi_net.init_layer(1, node_count=200, seq_node_cnt=wvecs_len)
  mozhi_net.init_layer(2, node_count=100, seq_node_cnt=wvecs_len)
  mozhi_net.init_layer(3, node_count=200, seq_node_cnt=0)
  mozhi_net.init_layer(4, node_count=wvecs_len, seq_node_cnt=0)
  mozhi_net.init_layer(5, node_count=wvecs_len, seq_node_cnt=0)
  mozhi_net.init_layer(6, node_count=wvecs_len, seq_node_cnt=0)
  
  
  if retrain:
    # Retrain the neural network with the training data received.
    if reenc:
      # Reencode the words into their feature vector representation.
      tr_wvecs_en, tr_vocab_en, tr_cc_mat_en = w2v.word2vec(tr_fpath+"eng.txt", win_size=1)
      tr_wvecs_fr, tr_vocab_fr, tr_cc_mat_fr = w2v.word2vec(tr_fpath+"frn.txt", win_size=1)
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
    
    # Getting the training data matrix for training
    tr_data_mat, tr_sent_len_en_list, tr_sent_len_fr_list = text2mat(tr_fpath, 
                                                                     tr_wvecs_en, 
                                                                     tr_vocab_en, 
                                                                     tr_wvecs_fr, 
                                                                     tr_vocab_fr,
                                                                     phr_wrd_cnt,
                                                                     wvecs_len)
      
    # Set training data to be used
    mozhi_net.set_train_data(tr_data_mat)
    # Searches for optimal weights using stochastic gradient descent
    mozhi_net.fit()
    pkl.dump(mozhi_net,open("models/mozhi_net.pkl","wb")) #Saves the entire object into a file
    
  else:
    mozhi_net = pkl.load(open("models/mozhi_net.pkl"))
    # Here we need to retreive all the data from pkl and assign it to the object
    # so that it can be used for testing (like weights).

    
  # Testing
  # This list will contain the equivalent french sentences.
  tst_target_fr = []
  
  # Getting the testing data matrix for predicting
  tst_data_mat,tst_sent_len_en_list, tst_sent_len_fr_list = text2mat(tst_fpath, 
                                                                     tr_wvecs_en, 
                                                                     tr_vocab_en, 
                                                                     tr_wvecs_fr, 
                                                                     tr_vocab_fr,
                                                                     phr_wrd_cnt,
                                                                     wvecs_len)
  # Remeber that this matrix will contain both the english and equivalent french
  
  tst_data_mat_en = tst_data_mat[:,0:wvecs_len] #Matrix containing only the english part
  
  #Matrix containing only the french part. This matrix can be used to compute accuracy later, maybe?
  tst_data_mat_fr = tst_data_mat[:,wvecs_len:]
  
  #This will be the matrix containing the predictions
  tst_target_mat_fr = np.zeros((1,wvecs_len))
  
  #Predicting
  for j in np.arange(0,tst_data_mat_en.shape[0]/phr_wrd_cnt):
    ii = j*phr_wrd_cnt
    out_fr = mozhi_net.predict(tst_data_mat_en[ii:ii+phr_wrd_cnt,:])
    tst_target_mat_fr = np.concatenate((tst_target_mat_fr,out_fr),axis=0)
  
  # We want to remove the first row which was zeros used for initialization
  tst_target_mat_fr = tst_target_mat_fr[1:,:] 
  
  # temporary variables help in storing the vector representation before
  # appending to the main list.
  temp1_list = [] 
  
  spidx = 0
  
  # Decoder
  for sidx in np.arange(0,len(tst_sent_len_fr_list)):
    temp_mat = tst_target_mat_fr[spidx:spidx+tst_sent_len_fr_list[sidx][1],:]
    for i in np.arange(0,tst_sent_len_fr_list[sidx][0]):
      temp1 = v2w.vec2word(in_vec = temp_mat[i,:], vocab = tr_vocab_fr,
                           word2vec_arr = tr_wvecs_fr)
      temp1_list.append(temp1)
    tst_target_fr.append(temp1_list)
    temp1_list = [] # cleaning the list for next iteration
    spidx = spidx+tst_sent_len_fr_list[sidx][1]
  
  
  tst_target_fr = [" ".join(word_list) for word_list in tst_target_fr]
  out_fname = "results/translated.txt"
  with open(out_fname, "wb") as res_file:
    [res_file.writelines(line+"\r\n") for line in tst_target_fr]
  res_file.close()
  
  return

 


if __name__ == "__main__":


  if platform.system() == "Linux":
    train_test_mozhi_net(tr_fpath="data/akns/set_01/",
                         tst_fpath="data/akns/set_01/",
                         retrain=True, 
                         reenc=True,
                         wvecs_len=8,
                         phr_wrd_cnt=3,
                         wvecs_en_fpath="data/wvecs_en_file.npy",
                         vocab_en_fpath="data/vocab_en_file.npy",
                         cc_mat_en_fpath="data/cc_mat_en_file.npy",
                         wvecs_fr_fpath="data/wvecs_fr_file.npy",
                         vocab_fr_fpath="data/vocab_fr_file.npy",
                         cc_mat_fr_fpath="data/cc_mat_fr_file.npy")
    # train_test_mozhi_net_for_debugging()

  else:
    train_test_mozhi_net(tr_fpath="data\\akns\\set_01\\", 
                         tst_fpath="data\\akns\\set_01\\",
                         retrain=True, 
                         reenc=True,
                         wvecs_len=8,
                         phr_wrd_cnt=3,
                         wvecs_en_fpath="data\\wvecs_en_file.npy",
                         vocab_en_fpath="data\\vocab_en_file.npy",
                         cc_mat_en_fpath="data\\cc_mat_en_file.npy",
                         wvecs_fr_fpath="data\\wvecs_fr_file.npy",
                         vocab_fr_fpath="data\\vocab_fr_file.npy",
                         cc_mat_fr_fpath="data\\cc_mat_fr_file.npy")
    # train_test_mozhi_net_for_debugging()
    
