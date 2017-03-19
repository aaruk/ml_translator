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
  dist = abs(in_vec - word2vec_arr) # Find Euclidean distance between input
  dist = dist * dist                # vector and all vectors in dictionary
  dist = dist.sum(axis=1)
  dist = np.sqrt(dist)
  word_idx = np.argmin(dist)

  # If distance is within acceptable limits, pull word from vocabulary
  if dist[word_idx] < 0.1:
    word = vocab[word_idx]
  else:
    word = "UNK"
  return word

if __name__ == "__main__":
  word_vec =  np.load("data/word_vec.npy")
  vocab  = np.load("data/vocab.npy")
  word_list = []
  for i in np.arange(word_vec.shape[0]):
    word_list.append(vec2word(word_vec[i], vocab, word_vec))
  print vocab
  print word_list
