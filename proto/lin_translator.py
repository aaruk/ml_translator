import proto.vec2word as vw
import numpy as np
import json

def read_word_vecs(fpath=""):
  """
  Reads words with their correspoding vector representations stored as space
  separated values from a file

  Input: Path to file containing word vecs
  ------
  Output: Matrix of word vecs, vocab dictionary
  ------
  """
  with open(fpath, 'r') as f:
    temp = f.readlines()
  f.close()

  temp = [x.strip("\n") for x in temp]
  temp_list = []
  elem = temp[0].split(' ')
  wvecs =  np.zeros((len(temp),len(elem)-1))
  vocab = {}
  for ii in np.arange(0, len(temp)):
    temp_list.append(temp[ii].split(" "))
    vocab[temp_list[ii][0]] = ii
    for jj in np.arange(1, len(elem)-1):
      wvecs[ii][jj-1] = temp_list[ii][jj]

  return wvecs, vocab


def train_wt(tr_fpath="", tst_fpath=""):
  """
  Learns a linear transformation between src and dst languages
  """

  # Read word vectors from a dictionary
  #src_wvecs, src_vocab = read_word_vecs(tr_fpath+"EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
  #dst_wvecs, dst_vocab = read_word_vecs(tr_fpath+"IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")

  # Load training data: word vectors & dictionary
  tr_fprefix = "datasets/europarl/"
  src_wvecs = np.load(tr_fprefix+"eng_wvecs_akns_w5_150.npy")
  dst_wvecs = np.load(tr_fprefix+"itl_wvecs_akns_w5_150.npy")

  eng_vocab, ital_vocab = {}, {}
  with open(tr_fprefix+"eng_vocab_akns_w5_150.json", "rb") as mf:
    src_vocab = json.load(mf)
  mf.close()
  with open(tr_fprefix+"itl_vocab_akns_w5_150.json", "rb") as mf:
    dst_vocab = json.load(mf)
  mf.close()

  src_wvec_len, dst_wvec_len = 150, 140
  # Initialize linear transformation matrix with random values
  wt = np.random.uniform(low=-0.5, high=0.5, size=(dst_wvec_len, src_wvec_len))

  # Read train_data
  with open(tr_fpath+"OPUS_en_it_europarl_train_5K.txt", 'r') as mf:
    tr_data = mf.readlines()
  mf.close()

  with open(tr_fprefix+"tr_eng.txt", 'r') as mf:
    src_words = mf.readlines()
  mf.close()
  with open(tr_fprefix+"tr_ital.txt", 'r') as mf:
    dst_words = mf.readlines()
  mf.close()
  dst_words = [d.strip('\n') for d in dst_words]
  src_words = [s.strip('\n') for s in src_words]

  # Take subset of tr_data
  #tr_data = tr_data[:500]
  #src_words, dst_words = [], []
  #for data in tr_data:
  #  data = data.strip("\n")
  #  eng, ital = data.split(" ")
  #  if eng in src_vocab and ital in dst_vocab:
  #    src_words.append(eng)
  #    dst_words.append(ital)

  sample_count = len(src_words)

  epochs = 30
  eta = 0.01
  # Stochastic gradient descent
  #NOTE: d1->src_dim, d2->dst_dim
  nf = 0
  for e in np.arange(epochs):
    found = 0
    for i in np.arange(sample_count):
      # Get source and dst word vec representations
      eng, ital = src_words[i], dst_words[i]
      if eng in src_vocab and ital in dst_vocab:
        src_word_ind = src_vocab[src_words[i]]
        x = src_wvecs[src_word_ind, :src_wvec_len] # d1,
        x = np.reshape(x, (x.shape[0], 1))

        dst_word_ind = dst_vocab[dst_words[i].decode('utf-8')]
        target = dst_wvecs[dst_word_ind, :dst_wvec_len] # 1xd2
        target = np.reshape(target, (target.shape[0], 1))

        # Transform src to dst vec representation
        pred = np.matmul(wt, x)  # d2x1

        # Backprop error due to transformation
        err = target - pred  # d2x1
        dw = np.matmul(err,  x.T)  # d2xd1
        wt = wt+eta*dw # d2xd1
    if e%5 == 0:
      np.savetxt("models/wt_mat_"+str(e)+".txt", wt)

    print "Epoch : ", e, "  "
    test_model(wt, src_words, src_wvecs, src_vocab, dst_words, dst_wvecs, dst_vocab)

  print "Not found = ",  nf
  tst_preds = test_model(wt, src_words, src_wvecs, src_vocab, dst_words, dst_wvecs, dst_vocab)
  with open('results/translated.json', 'wb') as mf:
    my_str= json.dumps(tst_preds, encoding="utf-8")
    mf.writelines(my_str)
  mf.close()
  return


def test_model(wts, src_words, src_wvecs, src_vocab, dst_words, dst_wvecs, dst_vocab):
  """
  """
  src_wvec_len, dst_wvec_len = 150, 140
  pred_dict = {}
  acc_list = []
  for src, dst in zip(src_words, dst_words):
    if src in src_vocab and dst in dst_vocab:
      src_ind = src_vocab[src] 
      x = src_wvecs[src_ind, :src_wvec_len]
      x = np.reshape(x, (x.shape[0], 1))
      dst_ind = dst_vocab[dst.decode("utf-8")]
      target = dst_wvecs[dst_ind, :dst_wvec_len]
      pred = np.matmul(wts, x)
      pred_t = np.reshape(pred, (pred.shape[0],))
      pred_word = vw.vec2word(pred_t, dst_vocab, dst_wvecs[:, :dst_wvec_len])  
      pred_dict[src] = (pred_word, dst) #.decode('utf-8')
      acc_list.append(pred_word == dst) #.decode('utf-8'))

  acc = np.array(acc_list).astype(np.uint8)
  #acc = acc_list.astype(np.uint8)
  acc = acc.sum() / len(dst_words)
  print "Accuracy ", acc*100, " %"
  return pred_dict


if __name__ == "__main__":
  tr_fpath = "datasets/transmat/data/"
  tst_fpath = "datasets/transmat/data/"
  train_wt(tr_fpath, tst_fpath)
