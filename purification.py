import glob
import os
import re


def purificate(text):
    result = text.replace(r'\n','\n')
    result = result.replace(r'\xa0', ' ')

    begin_end_regex = re.compile(r"\'\*\':\s*?(?P<article>.*)==",re.M | re.S)
    url_regex = re.compile('https?[\S]*')
    wiki_link_regex = re.compile('\[\[.*?\|(?P<word>[^|]*?)\]\]')
    # curved_bracers_reg = re.compile('\{\{.*?\}\}',re.M | re.S)
    # wiki_short_link = re.compile('\[\[(?P<link>[\w-–—\s]+?)\]\]')

    match = begin_end_regex.search(result)
    result = match.group('article')

    result = url_regex.sub('',result)

    #result = re.sub(curved_bracers_reg, '', result)
    # match = re.finditer(wiki_link_regex, result)
    # for gr in match:
    #     print (gr)
    # re.sub()

    punctuation_regex = re.compile('[^\w\s]')
    result = re.sub(punctuation_regex,'',result)

    return result

dirty_path = 'статьи необработанные'
pure_path = 'статьи'


for filename in glob.glob(os.path.join(dirty_path, '*.txt')):
    dirty_f = open(filename, mode='r')
    dirty_text = dirty_f.read()
    print(filename)
    pure_text = purificate(dirty_text)

    pure_f = open(os.path.join(pure_path, os.path.basename(filename)),'w')
    pure_f.write(pure_text)