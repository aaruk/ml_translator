""" Module with functions to convert dense vector representations to
    corresponding language words
"""
import numpy as np


def vec2word(in_vec=None, word2veclist=None, out_word=""):
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
  dist = abs(in_vec - word2veclist) # Find Euclidean distance between input
  dist = dist * dist                # vector and all vectors in dictionary
  word_idx = np.argmin(dist)

  # If distance is within acceptable limits, 
  if dist[word_idx] < 0.1:
    word = word_list[word_idx]
  else:
    word = "UNK"
  return word
