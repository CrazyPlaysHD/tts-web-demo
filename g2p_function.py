from unidecode import unidecode
import hparams as hp
import os
dict_w2p = {}
with open(os.path.join(hp.checkpoint_path, 'dict_phone2vn.txt'), 'r', encoding='utf-8') as rf:
    lines = rf.read().split('\n')
    for line in lines:
        temp = line.split()
        dict_w2p[temp[1]] = temp[0]

phu_am = ['b', 'c', 'ch', 'd', 'đ', 'g', 'h', 'gi', 'gh',  'k', 'kh', 'l', 'm', 'n',
          'ng', 'ngh', 'nh', 'p', 'ph', 'r', 's', 't', 'th', 'tr', 'v', 'x', 'q']

list_splited_p = ['ia', 'oo', 'ue', 'ye', 'yeu']

list_non_p = ['ạu', 'ảu', 'oẽ', 'oắ', 'oặ', 'oằ', 'oă', 'oẵ', 'oẳ', 'uơ', 'uọ', 'uớ', 'uờ', 'uở', 'êu']
dict_non_p = {}
for line in list_non_p:
    tmp = list(line)
    tmp2 = [dict_w2p[i] for i in tmp]
    new_line = ' '.join(tmp2)
    dict_non_p[line] = new_line


dict_nsp = {
    'ươu': 'ua2_T1 u1_T1',
    'ướu': 'ua2_T3 u1_T1',
    'ưỡu': 'ua2_T5 u1_T1',
    'ườu': 'ua2_T2 u1_T1',
    'uyê': 'uy_T1 e2_T1',
    'uyề': 'uy_T1 e2_T2',
    'uyế': 'uy_T1 e2_T3',
    'uyể': 'uy_T1 e2_T4',
    'uyễ': 'uy_T1 e2_T5',
    'uyệ': 'uy_T1 e2_T6',
    'oay': 'oa_T1 i_T1',
    'oáy': 'oa_T3 i_T1',
    'oảy': 'oa_T4 i_T1',
    'oạy': 'oa_T6 i_T1',
    'oeo': 'oe_T1 o1_T1',
    'oèo': 'oe_T2 o1_T1',
    'oéo': 'oe_T3 o1_T1',
    'oẻo': 'oe_T4 o1_T1',
    'oẹo': 'oe_T6 o1_T1',
    'uya': 'uy_T1 a1_T1',
    'uỳa': 'uy_T1 a1_T2',
    'uýa': 'uy_T1 a1_T3',
    'uỷa': 'uy_T1 a1_T4',
    'uỹa': 'uy_T1 a1_T5',
    'uỵa': 'uy_T1 a1_T6',
    'uay': 'ua_T1 i_T1',
    'uày': 'ua_T2 i_T1',
    'uáy': 'ua_T3 i_T1',
    'uây': 'u1_T1 ay3_T1',
    'uầy': 'u1_T1 ay3_T2',
    'uấy': 'u1_T1 ay3_T3',
    'uẩy': 'u1_T1 ay3_T4',
    'uẫy': 'u1_T1 ay3_T5',
    'uậy': 'u1_T1 ay3_T6',
    'uýu': 'uy_T3 u1_T1',
    'uỷu': 'uy_T4 u1_T1',
    'uỵu': 'uy_T6 u1_T1',
    'oao': 'o1_T1 ao_T1',
    'oáo': 'o1_T1 ao_T3',
    'uai': 'u1_T1 ai_T1',
    'uài': 'u1_T1 ai_T2',
    'uái': 'u1_T1 ai_T3',
    'uải': 'u1_T1 ai_T4',
    'uại': 'u1_T1 ai_T6',
    'ueo': 'u1_T1 eo_T1',
    'uèo': 'u1_T1 eo_T2',
    'uéo': 'u1_T1 eo_T3',
    'uẹo': 'u1_T1 eo_T6',
    'uă': 'u1_T1 a2_T1',
    'uằ': 'u1_T1 a2_T2',
    'uắ': 'u1_T1 a2_T3',
    'uẳ': 'u1_T1 a2_T4',
    'uẵ': 'u1_T1 a2_T5',
    'uặ': 'u1_T1 a2_T6',
    'uào': 'u1_T1 ao_T2',
    'uàu': 'u1_T1 au_T2',
    'uạu': 'u1_T1 au_T6',
    'uều': 'u1_T1 eu_T2',
    'oai': 'oa_T1 i_T1',
    'oài': 'oa_T2 i_T1',
    'oái': 'oa_T3 i_T1',
    'oải': 'oa_T4 i_T1',
    'oãi': 'oa_T5 i_T1',
    'oại': 'oa_T6 i_T1',
    'iễu': 'ie2_T5 u1_T1',
    'uồi': 'ua2_T2 i_T1',
    'ượi': 'ua2_T6 i_T1',
}

list_am = []
list_result = []
list_error_word = []


def word2p(line):
    word = '' + line.strip()
    line = line.strip()
    n = len(line)
    pre = ''
    if line.startswith('ngh'):
        pre = line[:3]
        line = line[3:]
    elif n >= 2 and line[:2] in ['ch', 'gi', 'kh', 'ng', 'ph', 'th', 'tr', 'nh', 'gh']:
        pre = line[:2]
        line = line[2:]
    elif n >= 1 and line[0] in phu_am:
        pre = line[:1]
        line = line[1:]

    n = len(line)
    end = ''
    if n >= 2 and line[-2:] in ['ch', 'gi', 'kh', 'ng', 'ph', 'th', 'tr', 'nh', 'gh']:
        end = line[-2:]
        line = line[:-2]
    elif n >= 1 and line[-1] in phu_am:
        end = line[-1:]
        line = line[:-1]

    if pre == '':
        new_pre = pre
    else:
        new_pre = dict_w2p[pre]

    if end == '':
        new_end = end
    else:
        new_end = dict_w2p[end]

    if line == '':
        return new_pre + ' ' + new_end, True

    # new_word = pre + line + end
    if line in dict_w2p:
        new_word = new_pre + ' ' + dict_w2p[line] + ' ' + new_end
        return new_word, True
    else:
        if line in dict_nsp:
            new_line = dict_nsp[line]
            new_word = new_pre + ' ' + new_line + ' ' + new_end
            return new_word, True
        elif unidecode(line) in list_splited_p:
            tmp = list(line)
            tmp2 = [dict_w2p[i] for i in tmp]
            new_line = ' '.join(tmp2)
            new_word = new_pre + ' ' + new_line + ' ' + new_end
            return new_word, True
        elif line in dict_non_p:
            new_line = dict_non_p[line]
            new_word = new_pre + ' ' + new_line + ' ' + new_end
            return new_word, True
        list_error_word.append(word)
        # print(list_error_word)
        return word, False



if __name__ == '__main__':
    print(word2p('â'))


