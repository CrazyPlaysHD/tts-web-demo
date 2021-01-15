""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text_fs2 input to the model.

The default is a set of ASCII characters that works well for English or text_fs2 that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text_fs2 import cmudict

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters  = 'aáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz'
_arpabet = ['@' + s for s in cmudict.valid_symbols]


# for Phoneme Version
# _silences = ['@sp', '@spn', '@sil']
# symbols = [_pad] + list(_special) + list(_punctuation) + _arpabet + _silences


# for Char Version
_silences = ['@sp']
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _silences


