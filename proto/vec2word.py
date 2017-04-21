""" Module with functions to convert dense vector representations to
    corresponding language words
"""
import numpy as np


def vec2word(in_vec=None, vocab=None, word2vec_arr=None, out_word=""):
  """
  Function takes a k-D vector(point) and finds closest match from a point
  cluster

  Input:
  ------
  in_vec        : k-D vector to be converted to word
  word_list     : ordered list of words corresponding to vectors in word2vec
  word2vec_list : list of vectors representing all words in vocabulary
  out_word      : word corresponding to input vector
  """
  sim_measure = "cosine"
  if sim_measure == "euclidean":
    dist = abs(in_vec - word2vec_arr) # Find Euclidean distance between input
    dist = dist * dist                # vector and all vectors in dictionary
    dist = dist.sum(axis=1)
    dist = np.sqrt(dist)
    word_idx = np.argmin(dist)
  elif sim_measure == "cosine":
    inner_prod = np.matmul(word2vec_arr, in_vec)
    vec_norm = word2vec_arr*word2vec_arr
    vec_norm = np.sqrt(vec_norm.sum(axis=1))
    in_vec_norm = in_vec*in_vec
    in_vec_norm = np.sqrt(in_vec_norm.sum())
    denom = in_vec_norm*vec_norm
    cosine_dist = inner_prod/(in_vec_norm*vec_norm)
    word_idx = np.argmax(cosine_dist)

  # If distance is within acceptable limits, pull word from vocabulary
  # TODO: Add code in previous modules to get a dictionary with
  #       Indices as keys and words as values
  word = vocab.keys()[vocab.values().index(word_idx)]
  return word

if __name__ == "__main__":
  word_vec =  np.load("data/word_vec.npy")
  vocab  = np.load("data/vocab.npy")
  word_list = []
  for i in np.arange(word_vec.shape[0]):
    word_list.append(vec2word(word_vec[i], vocab, word_vec))
  print vocab
  print word_list
