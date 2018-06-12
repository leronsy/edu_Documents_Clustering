import glob
import os
from pymystem3 import Mystem
import pymorphy2
import nltk
from nltk.corpus import brown


dirty_path = 'статьи'
pure_path = 'слова из статей'
functors_pos = {'INTJ', 'PRCL', 'CONJ', 'PREP'}
morth = pymorphy2.MorphAnalyzer()
m = Mystem()


def lem(text):
    words = m.lemmatize(text)
    return words


def pos(word):
    "Return a likely part of speech for the *word*."""
    return morth.parse(word)[0].tag.POS

stop_words= nltk.corpus.stopwords.words('russian')
print(type(stop_words))

for filename in glob.glob(os.path.join(dirty_path, '*.txt')):
    dirty_f = open(filename, mode='r')
    text = dirty_f.read()
    print(filename)
    text = text.replace('\n', ' ')
    words = lem(text)
    words = [w for w in words if ' ' not in w and pos(w) not in functors_pos and w not in stop_words]
    out = ' '.join(words)
    pure_f = open(os.path.join(pure_path, os.path.basename(filename)), 'w')
    pure_f.write(str(out))

