import json
from urllib.request import urlopen
from urllib.parse import quote
import os
titles = list()
with open('заголовки статей.txt', mode='r') as tf:
    a = 1
    while a:
        a = tf.readline()
        if len(a) > 1:
            titles.append(a.replace('\n', ''))
print(titles)

url_base = 'https://ru.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&&titles='
path = 'статьи необработанные'
for title in titles:
    coded_title = title.replace(' ', '_').encode('utf8')
    url = url_base+quote(coded_title)
    jsonurl = urlopen(url)
    web_file = json.loads(jsonurl.read())
    article = str(web_file)
    dirty_file = open(os.path.join(path, title+'.txt'),'w')
    dirty_file.write(article)
