import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

modifiers = [ '\u093a',
            '\u093b',
            '\u093c',
            '\u093d',#half
            '\u093e', #a
             '\u093f','\u0940', #e
             '\u0941','\u0942', #u
             '\u0943','\u0944',  #r
             '\u0945','\u0946',
               '\u0947','\u0948', #i
             '\u0949',
              '\u094a', '\u094b','\u094c',#o
             #'\u094d',
             '\u094e','\u094f',
            '\u0902','\u0903']
escapers = ['\u094d']

def isModifier(item):
    return item in modifiers
def getPhonemesList(word,i,ph):
    #print(word, i, ph)
    if i >= len(word):
        if len(word) >0:
            ph.append(word)
        #print(i, 'is greater,returning')
        return ph
    if word[i] in escapers:
        #print(i, 'was escaper, escaping', word[i])
        return getPhonemesList(word,i+1,ph);
    if isModifier(word[i]):
        if i+1 < len(word) and isModifier(word[i+1]):
            #print(i, 'next is modifier escaping', word[i])
            return getPhonemesList(word,i+1,ph);
        else:
            #print(i, 'is modifier splitting', word[:i+1])
            ph.append(word[:i+1])
        return getPhonemesList(word[i+1:],0,ph);
    else:
        if i > 0 and word[i-1] not in escapers:
            #print(i, 'was consonant before splitting', word[:i])
            ph.append(word[:i])
            return getPhonemesList(word[i:],0,ph);
        else:
            #print(i, 'is not modifier escaping', word[i])
            return getPhonemesList(word,i+1,ph);
    return ph
def getPh(word):
    arr = []
    return getPhonemesList(word,0,arr)

#number of words in devph.csv
num_words = 4181
maxlen = 30
# embed_dim = 150
# batch_size = 16
# nb_epoch = 30
filename='sm_bidi2.h5'
#tokenizer = Tokenizer(num_words = num_words, split=' ')
autoencoder = load_model(filename)
with open('tk.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
    
def isSanskritWord(word):
    sentence = ' '.join(getPh(word))
    seq = tokenizer.texts_to_sequences([sentence])
    pad_seq = pad_sequences(seq, maxlen)
    sentence_seq = autoencoder.predict(pad_seq)
    classes = list(np.argmax(sentence_seq[0], axis=1))
    predict_word = tokenizer.sequences_to_texts([classes])
    combined = "".join(predict_word);
    combined = combined.replace(' ', '');
    print(word, combined)
    return  combined == word