import glob
import os
import re


def purificate(text):
    result = text.replace(r'\n', '\n')
    result = result.replace(r'\xa0', ' ')
    result = result.replace("—","-").replace("–","-")
    begin_end_regex = re.compile(r"\'\*\':\s*?(?P<article>.*)==", re.M | re.S)
    url_regex = re.compile('http[\S]*')
    wiki_link_regex = re.compile(r'\[\[.*?\|(?P<word>[^\|]*?)\]\]', re.M | re.S)
    curved_brac_reg = re.compile('\{[^\{\}]*?\}', re.M | re.S)
    curved_brac2_reg = re.compile('\{\{[^\{\}]*?\}\}', re.M | re.S)
    html_tags_reg = re.compile('\<.*?\>', re.M | re.S)

    match = begin_end_regex.search(result)
    result = match.group('article')

    result = re.sub(curved_brac2_reg, '', result)
    result = re.sub(curved_brac2_reg, '', result)
    result = re.sub(curved_brac_reg, '', result)
    result = re.sub(html_tags_reg, '', result)
    result = url_regex.sub('', result)
    for match in wiki_link_regex.finditer(result):
        result = re.sub(match.re, match.group('word'), result, count=1)


    punctuation_regex = re.compile('[^-\w\s]')
    result = re.sub(punctuation_regex, '', result)

    return result


dirty_path = 'статьи необработанные'
pure_path = 'статьи'

for filename in glob.glob(os.path.join(dirty_path, '*.txt')):
    dirty_f = open(filename, mode='r')
    dirty_text = dirty_f.read()
    print(filename)
    pure_text = purificate(dirty_text)

    #print(pure_text)
    pure_f = open(os.path.join(pure_path, os.path.basename(filename)), 'w')
    pure_f.write(pure_text)
