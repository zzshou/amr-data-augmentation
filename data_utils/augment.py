import nlpaug.augmenter.word as naw
import nltk
from synonym import SynonymAug
from antonym import AntonymAug

from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm
import random

import re


def count_edge_in_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    edges = set()
    leaves = set()
    entities = set()
    for line in lines:
        items = line.split()
        for i, item in enumerate(items):
            if item.startswith(':') and not re.match(':ARG\d+|:wiki|:op\d+|:snt\d+|:value|:polarity', item):
                if items[i + 1] not in ('(', ')'):
                    edges.add(item)
                    leaves.add(' '.join([item, items[i + 1]]))
                elif items[i + 1] == '(' and items[i + 3] == ')':
                    edges.add(item)
                    leaves.add(' '.join([item, items[i + 1], items[i + 2], items[i + 3]]))
            elif item not in ('(', ')'):
                entities.add(item.lower())
    print('different entity: %d' % len(entities))
    print('different edge: %d' % len(edges))
    print('different leaves: %d' % len(leaves))
    # print(leaves)
    return list(entities), list(edges), list(leaves)


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

def replace_syntonym(line, n=1, alpha=0.1, vocab=None, aug=None):
    if not aug:
        aug = SynonymAug(aug_src='ppdb', model_path='/home/data/zshou/corpus/nltk/ppdb-2.0-tldr',
                         tokenizer=WhitespaceTokenizer(), vocab=vocab, aug_p=alpha)
    new_line = aug.augment(line, n=n)
    if n == 1:
        return [new_line]
    else:
        return new_line


def write_syntonym(input, output, alpha=0.1, n=1):
    # new_file = file.replace('.source', '_synaug_n%d.source' % n)
    new_lines = []
    aug = SynonymAug(aug_src='ppdb', model_path='/home/data/zshou/corpus/nltk/ppdb-2.0-tldr',
                     tokenizer=WhitespaceTokenizer(), aug_p=alpha)
    with open(input, 'r') as f:
        for line in tqdm(f):
            try:
                label, sentence = line.strip().split('\t')
                new_line = replace_syntonym(sentence, n=n, aug=aug)
                new_line = [label + '\t' + line for line in new_line]
            except:
                new_line = replace_syntonym(line.strip(), n=n, aug=aug)
            new_lines.append('\n'.join(new_line) + '\n')
    with open(output, 'w') as wf:
        wf.writelines(new_lines)

    print('replace syntonym finish')


### random deletion (leaves)
def _random_delete_leave(line):  # if graph has only one node, this function will do nothing
    is_leaf = False
    stack = []
    candidate = []
    leaves = []
    for item in line.split():
        if item.startswith(':') and not is_leaf:
            is_leaf = True
            candidate = [item]
        elif item.startswith(':') and len(stack) > 0:
            candidate = [item]
            stack = []
        elif item == '(' and is_leaf:
            candidate.append(item)
            stack.append(item)
        elif item == ')':
            if len(stack) == 1:
                stack = []
                candidate.append(')')
                leaves.append(' '.join(candidate))
                candidate = []
            elif len(stack) == 0 and len(candidate) > 0:
                leaves.append(' '.join(candidate))
                candidate = []
        elif is_leaf:
            candidate.append(item)
        else:
            continue
    if len(leaves) > 0:
        cand = random.choice(leaves)
        new_line = line.replace(cand, '').replace('\s+', ' ')
        # print(cand,new_line)
        return new_line
    else:
        return line


def random_delete_leaf(line, n=1, alpha=0.1):
    augs = []
    num_leaf = len([i for i in line if i.startswith(':')])
    while n > 0:
        new_line = line
        n_rd = max(1, int(alpha * num_leaf))
        while n_rd > 0:
            new_line = _random_delete_leave(new_line)
            n_rd -= 1
        augs.append(new_line)
        n -= 1
    return augs


def write_delete_leaf(input, output, alpha=0.1, n=1):
    # new_file = file.replace('.source', '_del_leafaug_n%d.source' % n)
    new_lines = []
    with open(input, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            label, sentence = line.strip().split('\t')
            new_line = random_delete_leaf(sentence, n, alpha)
            new_line = [label + '\t' + line for line in new_line]
        except:
            new_line = random_delete_leaf(line.strip(), n, alpha)
        new_lines.append('\n'.join(new_line) + '\n')
    with open(output, 'w') as wf:
        wf.writelines(new_lines)

    print('delete finish')


### random deletion (edge)
def _delete_edge(line: str, k: float = 0.9):
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


def delete_edge(line: str, k: float = 0.9, n=1):
    gs = set()
    try_count = 1
    while len(gs) < n and try_count < len(line.split()):
        try_count += 1
        ng = _delete_edge(line, k)
        if ng not in gs and ng != line:
            gs.add(ng)
    return list(gs)


def write_delete(file, k=0.9, n=1):
    new_file = file.replace('.source', '_delaug_n%d.source' % n)
    new_lines = []
    with open(file, 'r') as f:
        for line in tqdm(f):
            augs = delete_edge(line.strip(), k, n)
            new_lines.append('\n'.join(augs) + '\n')
    with open(new_file, 'w') as wf:
        wf.writelines(new_lines)


### random insertion (insert leaves from amr17)
def insert_edge(line, leaf_list, n=1, alpha=0.1):
    augs = []
    items = line.split(' ')
    num_edge = len([i for i in line if i.startswith(':')])
    while n > 0:
        new_items = items.copy()
        n_ri = max(1, int(alpha * num_edge))
        while n_ri > 0:
            new_leaf = random.choice(leaf_list)
            indexs = index_item(new_items, ')')
            ind = random.choice(indexs)
            new_items[ind:ind] = new_leaf.split()
            n_ri -= 1
        augs.append(' '.join(new_items))
        n -= 1
    return augs


def index_item(tokens, token):
    res = []
    for i, t in enumerate(tokens):
        if t == token:
            res.append(i)
    return res


def write_insert(input, output, leaf_list=None, leaf_file=None, alpha=0.1, n=1):
    if leaf_file:
        _, _, leaf_list = count_edge_in_file(leaf_file)
    elif leaf_list:
        leaf_list = leaf_list
    else:
        print('must input leaf_list or the file we can extract leaf')
        return
    # new_file = file.replace('.source', '_insertaug_n%d.source' % n)
    new_lines = []
    with open(input, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            label, sentence = line.strip().split('\t')
            new_line = insert_edge(sentence, leaf_list, n, alpha)
            new_line = [label + '\t' + line for line in new_line]
        except:
            new_line = insert_edge(line.strip(), leaf_list, n, alpha)
        new_lines.append('\n'.join(new_line) + '\n')
    with open(output, 'w') as wf:
        wf.writelines(new_lines)
    print('insert finish')


### random swap

def _swap_node(line):
    all_items = line.split()[1:-1]
    root = all_items.pop(0)
    if len(all_items) < 3:
        return line
    node_in_depth = {0: [root]}
    stack = ['(']
    edges = []

    for i, item in enumerate(all_items):
        if item.startswith(':'):
            edges.append([])
            if all_items[i + 1] != '(':
                stack.append('(')
        elif item == '(':
            stack.append(item)
        elif item == ')':
            if len(stack) > 0 and len(edges):
                stack.pop()
                last_edge = edges.pop(-1)
                last_edge.append(')')
                last_depth = len(stack)
            else:
                # print(line)
                continue
            if last_depth in node_in_depth.keys():
                node_in_depth[last_depth].append(last_edge)
            else:
                node_in_depth[last_depth] = [last_edge]
        elif all_items[i - 1].startswith(':'):
            if len(stack) > 0:
                stack.pop()
            last_edge = edges.pop(-1)
            last_edge.append(item)
            last_depth = len(stack)
            if last_depth in node_in_depth.keys():
                node_in_depth[last_depth].append(last_edge)
            else:
                node_in_depth[last_depth] = [last_edge]
        for i in edges:
            i.append(item)

    candidates = []
    for key, val in node_in_depth.items():
        if len(val) > 1:
            candidates.append(val)
    if len(candidates) < 1: return line
    cand = random.choice(candidates)
    l1, l2 = random.sample(cand, 2)
    p1 = ' '.join(l1)
    p2 = ' '.join(l2)
    line = line.replace(p1, '$wait_process$', 1)
    line = line.replace(p2, p1, 1)
    line = line.replace('$wait_process$', p2)
    return line
    # except:
    #     print(line)
    # return line


def swap_nodes(line, n=1, alpha=0.1):
    augs = []
    num_edge = len([i for i in line if i.startswith(':')])
    while n > 0:
        new_line = line
        n_rs = max(1, int(alpha * num_edge))
        while n_rs > 0:
            new_line = _swap_node(new_line)
            n_rs -= 1
        augs.append(new_line)
        n -= 1
    return augs


def write_swap(input, output, alpha=0.1, n=1):
    # new_file = file.replace('.source', '_swapaug_n%d.source' % n)
    new_lines = []
    with open(input, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            label, sentence = line.strip().split('\t')
            new_line = swap_nodes(sentence, n, alpha)
            new_line = [label + '\t' + line for line in new_line]
        except:
            new_line = swap_nodes(line.strip(), n, alpha)
        new_lines.append('\n'.join(new_line) + '\n')
    with open(output, 'w') as wf:
        wf.writelines(new_lines)

    print('swap finish')


### antonym replacement
# need to update

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


if __name__ == '__main__':
    entities, _, leaf_list = count_edge_in_file('../data/amr17/train.source')

    file = '../data/wiki-data/wiki-100000.source'
    sr = file.replace('.source', '_sr_0.1.source')
    write_syntonym(file, sr, alpha=0.05)
    rd = file.replace('.source', '_rd_0.1.source')
    write_delete_leaf(file, rd, 0.1)
    ins = file.replace('.source', '_insert_0.1.source')
    write_insert(file, ins, leaf_list, alpha=0.1)
