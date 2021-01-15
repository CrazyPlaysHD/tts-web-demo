""" from https://github.com/keithito/tacotron """

import re


# valid_symbols = [
#   'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
#   'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
#   'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
#   'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
#   'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
#   'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
#   'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
# ]

valid_symbols = ['a1_T1', 'a1_T2', 'a1_T3', 'a1_T4', 'a1_T5', 'a1_T6', 'a2_T1',
                 'a2_T2', 'a2_T3', 'a2_T4', 'a2_T5', 'a2_T6', 'a3_T1', 'a3_T2',
                 'a3_T3', 'a3_T4', 'a3_T5', 'a3_T6', 'ai_T1', 'ai_T2', 'ai_T3',
                 'ai_T4', 'ai_T5', 'ai_T6', 'ao_T1', 'ao_T2', 'ao_T3', 'ao_T4',
                 'ao_T5', 'ao_T6', 'au3_T1', 'au3_T2', 'au3_T3', 'au3_T4', 'au3_T5',
                 'au3_T6', 'au_T1', 'au_T2', 'au_T3', 'ay3_T1', 'ay3_T2', 'ay3_T3',
                 'ay3_T4', 'ay3_T5', 'ay3_T6', 'ay_T1', 'ay_T2', 'ay_T3', 'ay_T4',
                 'ay_T5', 'ay_T6', 'b', 'ch', 'd1', 'd2', 'e1_T1', 'e1_T2', 'e1_T3',
                 'e1_T4', 'e1_T5', 'e1_T6', 'e2_T1', 'e2_T2', 'e2_T3', 'e2_T4',
                 'e2_T5', 'e2_T6', 'eo_T1', 'eo_T2', 'eo_T3', 'eo_T4', 'eo_T5',
                 'eo_T6', 'eu_T2', 'eu_T3', 'eu_T4', 'eu_T6', 'g', 'h', 'i_T1',
                 'i_T2', 'i_T3', 'i_T4', 'i_T5', 'i_T6', 'ie2_T1', 'ie2_T2',
                 'ie2_T3', 'ie2_T4', 'ie2_T5', 'ie2_T6', 'ieu_T1', 'ieu_T2',
                 'ieu_T3', 'ieu_T4', 'ieu_T6', 'iu_T1', 'iu_T2', 'iu_T3', 'iu_T4',
                 'iu_T5', 'iu_T6', 'j', 'k', 'kh', 'l', 'm', 'n', 'ng', 'nh',
                 'o1_T1', 'o1_T2', 'o1_T3', 'o1_T4', 'o1_T5', 'o1_T6', 'o2_T1',
                 'o2_T2', 'o2_T3', 'o2_T4', 'o2_T5', 'o2_T6', 'o3_T1', 'o3_T2',
                 'o3_T3', 'o3_T4', 'o3_T5', 'o3_T6', 'oa_T1', 'oa_T2', 'oa_T3',
                 'oa_T4', 'oa_T5', 'oa_T6', 'oe_T1', 'oe_T2', 'oe_T3', 'oe_T4',
                 'oe_T6', 'oi2_T1', 'oi2_T2', 'oi2_T3', 'oi2_T4', 'oi2_T5',
                 'oi2_T6', 'oi3_T1', 'oi3_T2', 'oi3_T3', 'oi3_T4', 'oi3_T5',
                 'oi3_T6', 'oi_T1', 'oi_T2', 'oi_T3', 'oi_T4', 'oi_T5', 'oi_T6',
                 'p', 'ph', 'r', 's', 't', 'th', 'tr', 'u1_T1', 'u1_T2', 'u1_T3',
                 'u1_T4', 'u1_T5', 'u1_T6', 'u2_T1', 'u2_T2', 'u2_T3', 'u2_T4',
                 'u2_T5', 'u2_T6', 'ua2_T1', 'ua2_T2', 'ua2_T3', 'ua2_T4', 'ua2_T5',
                 'ua2_T6', 'ua_T1', 'ua_T2', 'ua_T3', 'ua_T4', 'ua_T5', 'ua_T6',
                 'ui2_T4', 'ui_T1', 'ui_T2', 'ui_T3', 'ui_T4', 'ui_T5', 'ui_T6',
                 'uoi2_T1', 'uoi2_T3', 'uoi2_T4', 'uoi2_T5', 'uoi2_T6', 'uoi3_T1',
                 'uoi3_T2', 'uoi3_T3', 'uoi3_T4', 'uoi3_T5', 'uou_T6', 'uu2_T1',
                 'uu2_T2', 'uu2_T3', 'uu2_T4', 'uu2_T5', 'uu2_T6', 'uy_T1', 'uy_T2',
                 'uy_T3', 'uy_T4', 'uy_T5', 'uy_T6', 'v', 'x']

_valid_symbol_set = set(valid_symbols)


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path, keep_ambiguous=True):
    if isinstance(file_or_path, str):
      with open(file_or_path, encoding='latin-1') as f:
        entries = _parse_cmudict(f)
    else:
      entries = _parse_cmudict(file_or_path)
    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries


  def __len__(self):
    return len(self._entries)


  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    return self._entries.get(word.upper())



_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  for part in parts:
    if part not in _valid_symbol_set:
      return None
  return ' '.join(parts)
