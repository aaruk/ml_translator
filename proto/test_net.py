import numpy as np
import json
import proto.neural_net as nn
import proto.word2vec as w2v
import proto.vec2word as v2w
import pickle as pkl
import pandas as pd
import ast

def text2mat(text_fpath=None, wvecs_en=None, vocab_en=None, wvecs_it=None, 
             vocab_it=None,
             phr_wrd_cnt=1,
             wvecs_len=8):
  """
  Converts english and italian sentences into a matrix with equivalent feature 
  vectors and padding.
  Input:
  ------
  text_fpath    : This should be the folder containing both the english and 
                  italian sentence text files.
  wvecs_en      : Feature vectors for English
  vocab_en      : Dictionary containing word indices for English
  wvecs_it      : Feature vectors for italian
  vocab_it      : Dictionary containing word indices for italian
  phr_wrd_cnt   : Number of words in the phrase when breaking the sentence
  wvecs_len     : This will decide the dimension of the vector representation
  """
  sent_en_list = [] #List of list which will contain english sentences
  sent_it_list = [] #List of list which will contain italian sentences
  
  # Creating a list of list for English
  with open(text_fpath+"eng.txt", 'r') as f:
    data_en = f.readlines()
  data_en = [x.strip("\n") for x in data_en]
  # Removing some additional characters it is adding to the file.
  data_en = [x.strip("\xef\xbb\xbf") for x in data_en]
  for i in np.arange(0, len(data_en)):
   sent_en_list.append(data_en[i].split(" "))
  
  # Creating a list of list for italian
  with open(text_fpath+"itl.txt", 'r') as f:
    data_it = f.readlines()
  data_it = [x.strip("\n") for x in data_it]
  # Removing some additional characters it is adding to the file.
  data_it = [x.strip("\xef\xbb\xbf") for x in data_it]
  for i in np.arange(0, len(data_it)):
   sent_it_list.append(data_it[i].split(" "))
   
  # Creating a list with padding included.
  wvecs_en_list = [] #This will be a list containing the feature vectors with padding
  wvecs_it_list = [] #This will be a list containing the feature vectors with padding
  
  #List which contains the length of sentences and length with padding
  sent_len_en_list = []
  sent_len_it_list = []
  
  padding = np.zeros(wvecs_len, dtype=np.float32)
  
  for s_idx in np.arange(0,len(sent_en_list)):
    # Since both english and italian will have the same number of sentences, I 
    # I am using just the english list length for the loop.
    sent_len_en = len(sent_en_list[s_idx]) # Sentence length(no. of words)
    sent_len_it = len(sent_it_list[s_idx]) # Sentence length(no. of words)
    phr_num_en = float(sent_len_en)/phr_wrd_cnt # Number of phrases the sentence can be broken into
    phr_num_en = np.ceil(phr_num_en)
    phr_num_it = float(sent_len_it)/phr_wrd_cnt
    phr_num_it = np.ceil(phr_num_it)
    sent_len_en_list.append((sent_len_en,int(phr_num_en*phr_wrd_cnt)))
    sent_len_it_list.append((sent_len_it,int(phr_num_it*phr_wrd_cnt)))
    if phr_num_en>=phr_num_it:
      phr_num_it = phr_num_en
    else:
      phr_num_en = phr_num_it
    
    for w_idx in np.arange(0,int(phr_num_en*phr_wrd_cnt)):
      # Could have used as well phr_num_it as both should be same with max value assigned
      if w_idx<sent_len_en:
        if sent_en_list[s_idx][w_idx].decode('utf-8') in vocab_en:
          # if the word can be found in vocabulary
          temp3 = wvecs_en[vocab_en[sent_en_list[s_idx][w_idx].decode('utf-8')],:]
        else:
          temp3 = padding
      else:
        temp3 = padding
      wvecs_en_list.append(temp3)
      
      if w_idx<sent_len_it:
        if sent_it_list[s_idx][w_idx].decode('utf-8') in vocab_it:
          # if the word can be found in vocabulary
          temp4 = wvecs_it[vocab_it[sent_it_list[s_idx][w_idx].decode('utf-8')],:]
        else:
          print "word not found"
          temp4 = padding
      else:
        temp4 = padding
      wvecs_it_list.append(temp4)
      
  wvecs_en_mat = np.array(wvecs_en_list) #Wordvec matrix for english with padding
  wvecs_it_mat = np.array(wvecs_it_list) #Wordvec matrix for italian with padding
  
  # This will be the matrix containing the vector representation of both
  # english and italian. With first wvecs_len columns being english and the
  # rest italian. This will also include the padding inbetween as needed.
  data_mat = np.zeros((len(wvecs_en_list),2*wvecs_len), dtype=np.float32)
  data_mat = np.concatenate((wvecs_en_mat, wvecs_it_mat), axis=1)
  
  return data_mat, sent_len_en_list, sent_len_it_list, sent_en_list, sent_it_list


def test_mozhi_net(tr_fpath=None, tst_fpath=None, retrain=True, 
                         reenc=False,
                         wvecs_len=8,
                         phr_wrd_cnt=1,
                         wvecs_en_fpath=None,
                         vocab_en_fpath=None,
                         cc_mat_en_fpath=None,
                         wvecs_it_fpath=None,
                         vocab_it_fpath=None,
                         cc_mat_it_fpath=None):
  """
  Train and test a given neural net architecture
  
  Input:
  ------
  tr_fpath       : training data file path, containing sentences. This should be 
                   the folder containing both the english and italian text files.
  tst_fpath      : testing data file path, conatining sentences
  retrain        : True if you want the neural network to be retrained
  reenc          : True if you want the words to be rencoded into vectors
  wvecs_len      : This will decide the dimension of the vector representation
  phr_wrd_cnt    : Number of words in the phrase when breaking the sentence
  wvecs_en_fpath : File path to be used to store the vector rep for english
  vocab_en_fpath : File path to be used to store the vocabulary for english
  cc_mat_en_fpath: File path to be used to store co occurrence matrix for eng 
  wvecs_it_fpath : File path to be used to store the vector rep for italian
  vocab_it_fpath : File path to be used to store the vocabulary for italian
  cc_mat_it_fpath: File path tone used to store co occurrence matrix for italian
   
  """
  # Initialize network architecture
  # say the phr_wrd_cnt=2, this means you will have 2 words as input. One going 
  # to inout layer and the second one going in parallel to the first hidden layer.
  # So seq_node_cnt=wvecs_len for layer1.
  mozhi_net = nn.NeuralNet(4, activation="tanh", max_epochs=500, eta=0.001, phr_wrd_cnt=phr_wrd_cnt)
  mozhi_net.init_layer(0, node_count=wvecs_len, seq_node_cnt=0, wts_fname="models/layer_wts_0_e30.txt")
  mozhi_net.init_layer(1, node_count=wvecs_len, seq_node_cnt=wvecs_len, wts_fname="models/layer_wts_1_e30.txt")
  mozhi_net.init_layer(2, node_count=wvecs_len, seq_node_cnt=wvecs_len, wts_fname="models/layer_wts_2_e30.txt")
  mozhi_net.init_layer(3, node_count=wvecs_len, seq_node_cnt=0, wts_fname="models/layer_wts_3_e30.txt")
  #mozhi_net.init_layer(4, node_count=wvecs_len, seq_node_cnt=0, wts_fname="models/layer_wts_4_e30.txt")
  #mozhi_net.init_layer(5, node_count=wvecs_len, seq_node_cnt=0,wts_fname="models/layer_wts_5_e30.txt")
  #mozhi_net.init_layer(6, node_count=wvecs_len, seq_node_cnt=0, wts_fname="models/layer_wts_6_e30.txt")
  
  
  if retrain:
    # Retrain the neural network with the training data received.
    if reenc:
      # Reencode the words into their feature vector representation.
      tr_wvecs_en, tr_vocab_en, tr_cc_mat_en = w2v.word2vec(tr_fpath+"eng.txt", win_size=3)
      tr_wvecs_it, tr_vocab_it, tr_cc_mat_it = w2v.word2vec(tr_fpath+"itl.txt", win_size=3)
      np.save(wvecs_en_fpath,tr_wvecs_en) # Saving the entire umatrix
      np.save(vocab_en_fpath,tr_vocab_en) 
      np.save(cc_mat_en_fpath,tr_cc_mat_en)
      np.save(wvecs_it_fpath,tr_wvecs_it)
      np.save(vocab_en_fpath,tr_vocab_it)
      np.save(cc_mat_it_fpath,tr_cc_mat_it)
      # Choose top wvecs_len singular vectors
      tr_wvecs_en = tr_wvecs_en[:,:wvecs_len]
      tr_wvecs_it = tr_wvecs_it[:,:wvecs_len]
      # with open("datasets\\eng_vocab.json", "wb") as mf:
        # my_str = json.dumps(tr_vocab_en, encoding="utf-8")
        # mf.writelines(my_str)
      # mf.close()
      # with open("datasets\\it_vocab.json", "wb") as mf:
        # my_str = json.dumps(tr_vocab_it, encoding="utf-8")
        # mf.writelines(my_str)
      # mf.close()
      # return
      # Plotting 3D figure
      # fig = plt.figure()
      # ax1 = fig.add_subplot(111, projection='3d')
      # for kk in np.arange(0,tr_wvecs_en.shape[0]):
        # ax1.scatter(tr_wvecs_en[kk,0],tr_wvecs_en[kk,1],tr_wvecs_en[kk,2])
        # ax1.text(tr_wvecs_en[kk,0],tr_wvecs_en[kk,1],tr_wvecs_en[kk,2],tr_vocab_en.keys()[tr_vocab_en.values().index(kk)])
      # plt.show()
      # fig = plt.figure()
      # ax2 = fig.add_subplot(111, projection='3d')
      # for kk in np.arange(0,tr_wvecs_it.shape[0]):
        # ax2.scatter(tr_wvecs_it[kk,0],tr_wvecs_it[kk,1],tr_wvecs_it[kk,2])
        # ax2.text(tr_wvecs_it[kk,0],tr_wvecs_it[kk,1],tr_wvecs_it[kk,2],tr_vocab_it.keys()[tr_vocab_it.values().index(kk)])
      # plt.show()
      
    else:
      # load feature vectors from encoding done previously from word2vec embedding
      # this part of the code for else needs to be checked.
      tr_wvecs_en = np.load(wvecs_en_fpath)
      with open(vocab_en_fpath) as mf:
        tr_vocab_en = json.loads(mf.read())
      mf.close()
      tr_wvecs_it = np.load(wvecs_it_fpath)
      with open(vocab_it_fpath) as mf:
        tr_vocab_it = json.loads(mf.read())
      mf.close()
      #Choose top wvecs_len singular vectors
      tr_wvecs_en = tr_wvecs_en[:,:wvecs_len]
      tr_wvecs_it = tr_wvecs_it[:,:wvecs_len]

    
    # Getting the training data matrix for training
    tr_data_mat, tr_sent_len_en_list, tr_sent_len_it_list, tr_sent_en_list, tr_sent_it_list = text2mat(tr_fpath, 
                                                                     tr_wvecs_en, 
                                                                     tr_vocab_en, 
                                                                     tr_wvecs_it, 
                                                                     tr_vocab_it,
                                                                     phr_wrd_cnt,
                                                                     wvecs_len)
      

    # Set training data to be used
    mozhi_net.set_train_data(tr_data_mat)
    # Searches for optimal weights using stochastic gradient descent
    mozhi_net.fit()
    pkl.dump(mozhi_net,open("models/mozhi_net.pkl","wb")) #Saves the entire object into a file
    
  else:
    # mozhi_net = pkl.load(open("models/mozhi_net.pkl"))
    # Here we need to retreive all the data from pkl and assign it to the object
    # so that it can be used for testing (like weights).
    
    
    tr_wvecs_en = np.load(wvecs_en_fpath)
    with open(vocab_en_fpath) as mf:
      tr_vocab_en = json.loads(mf.read())
    mf.close()
    tr_wvecs_it = np.load(wvecs_it_fpath)
    with open(vocab_it_fpath) as mf:
      tr_vocab_it = json.loads(mf.read())
    mf.close()
    #Choose top wvecs_len singular vectors
    tr_wvecs_en = tr_wvecs_en[:,:wvecs_len]
    tr_wvecs_it = tr_wvecs_it[:,:wvecs_len]

    
  # Testing
  # This list will contain the equivalent italian sentences.
  tst_target_it = []
  
  # Getting the testing data matrix for predicting
  tst_data_mat,tst_sent_len_en_list, tst_sent_len_it_list, tst_sent_en_list, tst_sent_it_list = text2mat(tst_fpath, 
                                                                     tr_wvecs_en, 
                                                                     tr_vocab_en, 
                                                                     tr_wvecs_it, 
                                                                     tr_vocab_it,
                                                                     phr_wrd_cnt,
                                                                     wvecs_len)
  # Remeber that this matrix will contain both the english and equivalent italian
  
  tst_data_mat_en = tst_data_mat[:,0:wvecs_len] #Matrix containing only the english part
  
  #Matrix containing only the italian part. This matrix can be used to compute accuracy later, maybe?
  tst_data_mat_it = tst_data_mat[:,wvecs_len:]
  
  #This will be the matrix containing the predictions
  tst_target_mat_it = np.zeros((1,wvecs_len))
  
  #Predicting
  for j in np.arange(0,tst_data_mat_en.shape[0]/phr_wrd_cnt):
    ii = j*phr_wrd_cnt
    out_it = mozhi_net.predict(tst_data_mat_en[ii:ii+phr_wrd_cnt,:])
    tst_target_mat_it = np.concatenate((tst_target_mat_it,out_it),axis=0)
  
  # We want to remove the first row which was zeros used for initialization
  tst_target_mat_it = tst_target_mat_it[1:,:] 
  
  # temporary variables help in storing the vector representation before
  # appending to the main list.
  temp1_list = [] 
  
  spidx = 0
  
  # Decoder
  for sidx in np.arange(0,len(tst_sent_len_it_list)):
    temp_mat = tst_target_mat_it[spidx:spidx+tst_sent_len_it_list[sidx][1],:]
    for i in np.arange(0,tst_sent_len_it_list[sidx][0]):
      temp1 = v2w.vec2word(in_vec = temp_mat[i,:], vocab = tr_vocab_it,
                           word2vec_arr = tr_wvecs_it)
      temp1_list.append(temp1)
    tst_target_it.append(temp1_list)
    temp1_list = [] # cleaning the list for next iteration
    spidx = spidx+tst_sent_len_it_list[sidx][1]
  
  print tst_target_it
  trans_dict = {}
  for en_sent,targ_sent,pred_sent in zip(tst_sent_en_list, tst_sent_it_list, tst_target_it):
    en_sent = " ".join(en_sent)
    targ_sent = " ".join(targ_sent)
    pred_sent = " ".join(pred_sent)
    trans_dict[en_sent] = {"target":targ_sent, "pred":pred_sent}
 
  with open("results/translated.json", "wb") as mf:
    my_str = json.dumps(trans_dict, encoding='utf-8')
    mf.writelines(my_str)
  mf.close()
  # tst_target_it = [" ".join(word_list) for word_list in tst_target_it]
  # out_fname = "results/translated.txt"
  # with open(out_fname, "wb") as res_file:
    # [res_file.writelines(line+"\r\n") for line in tst_target_it]
  # res_file.close()
  
  return

 


if __name__ == "__main__":
  test_mozhi_net(tr_fpath="datasets\\europarl\\", 
                       tst_fpath="datasets\\europarl\\",
                       retrain=False, 
                       reenc=False,
                       wvecs_len=180,
                       phr_wrd_cnt=3,
                       wvecs_en_fpath="datasets\\europarl\\eng_wvecs_akns_w5_150.npy",
                       vocab_en_fpath="datasets\\europarl\\eng_vocab_akns_w5_150.json",
                       wvecs_it_fpath="datasets\\europarl\\itl_wvecs_akns_w5_150.npy",
                       vocab_it_fpath="datasets\\europarl\\itl_vocab_akns_w5_150.json")
  
