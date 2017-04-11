"""
Module with functions to encode words into dense real-valued vectors
"""
import sys
import numpy as np


def word2vec(fpath="", win_size=1):
  """
  Reads text file and computes dense vectors using cooccurence matrix + svd
  """ 
  in_data = []
  # Read sentences line by line into list
  with open(fpath, 'r') as f:
    in_data = f.readlines()
  in_data = [x.strip("\n") for x in in_data]
  in_data = [x.strip("\xef\xbb\xbf") for x in data]

  vocab = " ".join(in_data)
  vocab = np.unique(vocab.split(" "))
  vocab_rows, vocab_cols = vocab.shape[0], vocab.shape[0]
  cc_mat = np.zeros((vocab_rows, vocab_cols), dtype=np.float32)

  # Store indices of words from vocab in a dictionary 
  vocab_indices = {}
  for i in np.arange(vocab_rows):
    vocab_indices[vocab[i]] = i

  #win_size = 1 # window size for cooccurence matrix
  # Go through each word in vocabulary and update cooccurence matrix
  for i, sentence in enumerate(in_data):
    word_list = sentence.split(" ")
    for wi, word in enumerate(word_list):
      wi_cc_ind = vocab_indices[word]  # Index of current word in vocab
      nbr_indices = []

      # Append valid indices of neighbors in a list
      for ni in np.arange(wi-win_size, wi+win_size+1):
        if ((ni != wi) and (ni >= 0) and (ni < len(word_list))):
          nbr_indices.append(ni)

      # Get indices of nbr words in vocabulary
      nbr_word_list = [word_list[ni] for ni in nbr_indices]
      nbr_cc_ind = [vocab_indices[w] for w in nbr_word_list]

      for nci in nbr_cc_ind:
        cc_mat[wi_cc_ind, nci] += 1
 
  #TODO: Compute SVD of cc_mat without np if required
  Umat, S, Vmat = np.linalg.svd(cc_mat)
  return Umat, vocab, cc_mat


if __name__ == "__main__":
  if len(sys.argv) > 1:
    fpath = sys.argv[1]
  else:
    fpath = "data/akns/set_00/eng.txt"
  word_vecs, vocab = word2vec(fpath)
  print word_vecs
  print vocab
