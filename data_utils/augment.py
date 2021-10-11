import re
import nlpaug.augmenter.word as naw
import nltk
from data_utils.synonym import SynonymAug
from data_utils.antonym import AntonymAug

# nltk.data.path.append('/Users/shouziyi/experiment/data/nltk_data')
nltk.data.path.append('/home/data/zshou/corpus/nltk')

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
import random


### polarity conversion

def convert_polarity(line):
    neg = ':polarity - :'
    if re.findall(':polarity -', line):  # delete polarity - in the original sentences
        new_line = re.sub(':polarity - ', '', line, 1)
    elif re.findall('^\( multi-sentence', line):  # multiple sentences
        if re.findall('(:snt\d+ \( and ):', line):
            new_line = re.sub('(:snt\d+ \( and :op1 \( \S+ ):', '\\1' + neg, line)
        else:
            new_line = re.sub('(:snt\d+ \( \S+ ):', '\\1' + neg, line)
    elif re.findall('^(\( \S+ ):', line):
        if re.findall('^(\( and ):', line):  # if begin with and
            new_line = re.sub('^(\( and :op1 \( \S+ ):', '\\1' + neg, line)
        else:
            new_line = re.sub('^(\( \S+ ):', '\\1' + neg, line)  # add negative after main predicate
    else:
        new_line = re.sub('\)\n', ':polarity - \)\n', line)
        # print(new_line)

    return new_line


def write_negative(file):
    new_file = file.replace('.source', '_neg.source')
    new_lines = []
    with open(file, 'r') as f:
        for line in f:
            new_line = convert_polarity(line)
            new_lines.append(new_line)
    with open(new_file, 'w') as wf:
        wf.writelines(new_lines)


### syntonym replacement

def replace_syntonym(line, n=1, aug=None):
    if not aug:
        aug = SynonymAug(tokenizer=WhitespaceTokenizer())
    new_line = aug.augment(line, n=n)
    return new_line


def write_syntonym(file):
    new_file = file.replace('.source', '_synaug.source')
    new_lines = []
    aug = SynonymAug(tokenizer=WhitespaceTokenizer())
    with open(file, 'r') as f:
        for line in tqdm(f):
            new_line = replace_syntonym(line, aug)
            new_lines.append(new_line + '\n')
    with open(new_file, 'w') as wf:
        wf.writelines(new_lines)


### antonym replacement

def replace_antonym(line, n=1, aug=None):
    if not aug:
        aug = AntonymAug(tokenizer=WhitespaceTokenizer())
    new_line = aug.augment(line, n=n)
    return new_line


def write_antonym(file):
    new_file = file.replace('.source', '_antaug.source')
    new_lines = []
    aug = AntonymAug(tokenizer=WhitespaceTokenizer())
    with open(file, 'r') as f:
        for line in tqdm(f):
            new_line = replace_antonym(line, aug)
            new_lines.append(new_line + '\n')
    with open(new_file, 'w') as wf:
        wf.writelines(new_lines)


### random deletion

def _delete_edge(line: str, k: float = 0.7):
    items = line.split(' ')
    new_line_items = []
    if_del = False
    brackets = []
    for item in items:
        if if_del:
            if item == '(':
                brackets.append(item)
            elif item == ')':
                if len(brackets) == 0:
                    new_line_items.append(item)
                    if_del = False
                else:
                    brackets.pop(-1)
                    if len(brackets) == 0:
                        if_del = False
        elif item.startswith(':'):
            if_del = random.uniform(0, 1) > k
            if not if_del:
                new_line_items.append(item)
        else:
            new_line_items.append(item)
    return ' '.join(new_line_items)


def delete_edge(line: str, k: float = 0.7, n=1):
    gs = set()
    try_count = 1
    while len(gs) < n and try_count < len(line.split()):
        try_count += 1
        ng = _delete_edge(line, k)
        if ng not in gs and ng != line:
            gs.add(ng)
    return list(gs)


def write_delete(file, k=0.7):
    new_file = file.replace('.source', '_delaug.source')
    new_lines = []
    with open(file, 'r') as f:
        for line in tqdm(f):
            new_line = delete_edge(line, k)
            new_lines.append(new_line)
    with open(new_file, 'w') as wf:
        wf.writelines(new_lines)


### random insertion

def insert_edge(line, n=1):
    node_list = ['test_node','test_node2']
    edge_list = [':test_edge']
    augs = []
    items = line.split(' ')
    while n > 0:
        new_edge = random.choice(edge_list)
        new_node = random.choice(node_list)
        indexs = index_item(items, ')')
        ind = random.choice(indexs)
        items.insert(ind, new_edge)
        items.insert(ind + 1, new_node)
        augs.append(' '.join(items))
        n -= 1

    return augs


def index_item(tokens, token):
    res = []
    for i, t in enumerate(tokens):
        if t == token:
            res.append(i)
    return res


### random swap

def swap_nodes(line, n=1):
    augs = []
    items = line.split(' ')

    return augs


### perspective rotation
def psp_rot(line, n=1):
    augs = []
    items = line.split(' ')

    return augs


if __name__ == '__main__':
    # file_name = 'wiki_amr_all.source'
    # write_delete(file_name)
    line = '( develop :ARG1 ( thing :ARG2-of ( name ) :ARG1-of ( local ) :quant ( many ) ) :ARG2 ( fruit ) :ARG2-of ( result :ARG1 ( cultivate :ARG1 fruit :ARG1-of ( spread :ARG1-of ( wide ) ) ) ) )'
    line = swap_nodes(line, n=5)
    print(line)

