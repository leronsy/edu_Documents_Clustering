import json
from urllib.request import urlopen
from urllib.parse import quote, urlencode
from flatten_dict import flatten
from pymystem3 import Mystem
import re

m = Mystem()
url_base = 'https://ru.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&&titles='
#url = 'https://ru.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&&titles=Рассказы_братьев_Стругацких'
url_title = 'Наполеон I'
# jsonurl = urlopen(url)
# file = json.loads(jsonurl.read())
# article = str(file)
# # for k, v in flatten(file).items():
# #     print('key:', k, '\n', v, type(v))
# print(article.replace(r'\n', '\n'))

print (quote(url_title.encode('utf8')))
# print(flatten(file))
# for k, v in
# article.replace('\\xa0',' ')
# with open('https://ru.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&&titles=%D0%A0%D0%B0%D1%81%D1%81%D0%BA%D0%B0%D0%B7%D1%8B_%D0%B1%D1%80%D0%B0%D1%82%D1%8C%D0%B5%D0%B2_%D0%A1%D1%82%D1%80%D1%83%D0%B3%D0%B0%D1%86%D0%BA%D0%B8%D1%85',encoding=('unicode-escape')) as f:
#        article = f.read()
#        #
#        #print(article.decode('latin1'))
# print(article)
# title_regex = re.compile(r'\"title\".*?\"(?P<title>.*?)\"')
# match = title_regex.search(article)
# title = match.group('title')
#
# print(title)
# begin_end_regex = re.compile(r'\{.*?$(?P<article>.*)==\sЛитература',re.M | re.S)
# match = begin_end_regex.search(article)
# article = match.group('article')
# print(article)
# wiki_link1 = re.compile('\[\[(\w+?)\]\]')
# wiki_link_regex = re.compile('(\[\[.*?\|)|(\{\{.*?\}\})')
#
# article = re.sub(wiki_link1, '', article)
# article = re.sub(wiki_link_regex, '', article)

# punctuation_regex = re.compile('[^\w\s]')
# article = re.sub(punctuation_regex,'',article)

# #
# print(article)
# lem_lst = m.lemmatize(t)
# word_lst = [w for w in lem_lst if not ' ' in w]
# print(word_lst)
