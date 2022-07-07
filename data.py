import glob

from khmernltk import *

from utils import *


def find_files(path):
    return glob.glob(path)


path = 'data/sleuk-rith'
all_articles = []
article = []

for file in find_files(f'{path}/*.txt'):
    article_list = read_from_txt(file)
    for lines in article_list:
        all_articles.append(lines)

lines_path = 'data/sleuk_rith_lines.txt'
with open(lines_path, 'w', encoding='utf-8') as new_file:
    for line in all_articles:
        new_file.write(line)

lines = read_from_txt(lines_path)

word_list = []
for line in lines:
    words_from_lines = word_tokenize(line)
    for word in words_from_lines:
        word_list.append(word.replace("<UNK>", ""))

filtered_list = list(dict.fromkeys(word_list))  # Remove duplicated words

words_path = 'data/sleuk_rith_words.txt'
with open(words_path, 'w', encoding='utf-8') as new_file:
    for word in filtered_list:
        new_file.write(word)
        new_file.write('\n')
