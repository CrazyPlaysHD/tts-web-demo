from collections import defaultdict
from g2p_function import word2p
from text_fs2.symbols import _punctuation
import argparse

class G2p:
    def __init__(self, path_dict):
        self.dict_word_char = defaultdict(list)
        self.dict_w2p = {}
        list_unuse = ['ieu_T5', 'uoi2_T2', 'uoi3_T6']
        with open(path_dict, 'r', encoding='utf8') as rf:
            lines = rf.read().split('\n')
            for line in lines:
                check = True
                for i in list_unuse:
                    if i in line:
                        check = False
                        break
                if not check:
                    continue
                temp = line.split()
                word = temp[0]
                phonme = ' '.join(temp[1:])
                self.dict_w2p[word] = phonme
                if word not in self.dict_word_char[word[0]]:
                    self.dict_word_char[word[0]].append(word)
        self.list_error = []

    def g2p(self, sentence):
        list_word = sentence.split()
        for i, word in enumerate(list_word):
            if word in self.dict_word_char[word[0]]:
                list_word[i] = self.dict_w2p[word]
                continue
            elif word in _punctuation:
                list_word[i] = 'sp'
            p, check = word2p(word)
            if check:
                list_word[i] = p
            else:
                list_word[i] = 'spn'
                self.list_error.append(p)
        return ' '.join(list_word)

    def make_dict_mfa(self, sentence):
        dict_tmp = {}
        list_word = sentence.split()
        for i, word in enumerate(list_word):
            if word in self.dict_word_char[word[0]]:
                dict_tmp[word] = self.dict_w2p[word]
                continue
            p, check = word2p(word)
            if check:
                dict_tmp[word] = p
            else:
                self.list_error.append(p)
        return dict_tmp


if __name__ == '__main__':
    gen = G2p('syllable_g2p.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', type=str, default='/content/metadata.csv')
    parser.add_argument('--dict_path', type=str, default='/content/dict-vne.txt')
    args = parser.parse_args()

    dict_vne = {}
    with open(args.meta_path, 'r', encoding='utf-8') as rf:
        lines = rf.read().split('\n')
        for i, line in enumerate(lines):
            tmp = line.split('|')
            text = tmp[-1]
            dict_vne.update(gen.make_dict_mfa(text))
            if i % 1000 == 0:
                print('Done', i)
    with open(args.dict_path, 'w', encoding='utf-8') as wf:
        for k, v in dict_vne.items():
            wf.write(k + ' ' + v + '\n')

    # print(gen.g2p('xin chào tùng xê tê moi x ki'))