import sys
import numpy as np
import json

import vec2word as vw


def translate(wt_mat="", eng=""):
  """
  Gets english sentence and returns italian sentence
  """
  src_vocab, dst_vocab = {}, {}
  src_wvecs = np.load("datasets/europarl/eng_wvecs_akns_w3_150.npy")
  dst_wvecs = np.load("datasets/europarl/itl_wvecs_akns_w3_150.npy")
  with open('datasets/europarl/eng_vocab_akns_150.json', 'r') as mf:
    src_vocab = json.load(mf)
  mf.close()
  with open('datasets/europarl/itl_vocab_akns_150.json', 'r') as mf:
    dst_vocab = json.load(mf)
  mf.close()

  src_words = eng.split(" ")
  #word_inds = [src_vocab[sw.encode('utf-8')] for sw in src_words]

  pred_list = []
  src_wvec_len = wt_mat.shape[1]
  dst_wvec_len = wt_mat.shape[0]
  for word in src_words:
    word =  word.lower()
    if word in src_vocab:
      wind = src_vocab[word.encode('utf-8')]
      x = src_wvecs[wind, :src_wvec_len]
      pred = np.matmul(wt_mat, x)
      pred_word = vw.vec2word(pred, dst_vocab, dst_wvecs[:, :dst_wvec_len])  
      pred_list.append(pred_word)
    else:
      pred_list.append("UNK")

  italian = " ".join(pred_list)
  print italian
    
  return  italian


if __name__ == "__main__":
  print "len ", len(sys.argv)
  if (len(sys.argv) > 1):
    eng = sys.argv[1]
    print eng
    wt_mat = np.loadtxt("lin_model/wt_mat_30.txt")
    translate(wt_mat,  eng)
  else:
    print "Syntax is ./proto/translate <eng_sent_to_translate>"
