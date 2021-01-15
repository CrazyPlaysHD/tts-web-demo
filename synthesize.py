import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from unidecode import unidecode

from fastspeech2.model_fs2 import FastSpeech2
from text_fs2 import text_to_sequence, sequence_to_text
import hparams as hp
import utils
from string import punctuation
from G2p import G2p
from time import time
import re
from evaluating_sentence import EvalMeanFrequencyScore

ev = EvalMeanFrequencyScore()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def preprocess(text, g2p):
    text = text.rstrip(punctuation).lower()
    print(text)
    phone = g2p.g2p(text)
    # phone = list(filter(lambda p: p != ' ', phone))
    # print(phone)

    phone = '{' + '}{'.join(phone.split()) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    # print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    # print(sequence_to_text(sequence))
    # print(sequence)
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)


def get_FastSpeech2():
    checkpoint_path = os.path.join(
        hp.restore_path, "checkpoint_{}.pth.tar".format('120000'))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, waveglow, text, idx, prefix='', duration_control=1.0, pitch_control=1.0, energy_control=1.0):
    t = time()
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control)

    if device == 'cuda':
        mel_postnet = mel_postnet.transpose(1, 2).detach()
    else:
        mel_postnet = mel_postnet.cpu().transpose(1, 2).detach()

    if not os.path.exists(args.test_path):
        os.makedirs(args.test_path)
    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
    #     hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, name)))
    t1 = time() - t
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet, waveglow, os.path.join(
            args.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, idx)))
    t2 = time() - t
    print('{}: time FS: {} (s) time {}: {}'.format(idx, t1, hp.vocoder, t2 - t1))


def gen_mel(model, text, idx, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control)
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_filename = '{}-mel.npy'.format(idx)
    np.save(os.path.join(args.test_path, mel_filename), mel, allow_pickle=False)
    mel_filename = '{}-mel-post.npy'.format(idx)
    np.save(os.path.join(args.test_path, mel_filename), mel_postnet, allow_pickle=False)


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='./test/')
    parser.add_argument('--path', type=str, default='test.txt')
    parser.add_argument('--dict_path', type=str, default='syllable_g2p.txt')
    parser.add_argument('--sent', type=str, default='?')
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)

    args = parser.parse_args()

    g2p = G2p(args.dict_path)

    sentences = []
    if args.sent != '?':
        sentences.append(args.sent)
    else:
        with open(args.path, 'r', encoding='utf-8') as rf:
            lines = rf.read().split('\n')
            for i, line in enumerate(lines):
                sentences.append(line)

    model = get_FastSpeech2().to(device)
    waveglow = utils.get_waveglow()
    # waveglow = None
    with torch.no_grad():
        for idx, sentence in enumerate(sentences):
            score = ev.cal(sentence)
        text = preprocess(sentence, g2p)
        name = str(idx) + '_' + str(score)
        print(text.shape)
        synthesize(model, waveglow, text, name, 'step_{}'.format(
            '120000'), args.duration_control, args.pitch_control, args.energy_control)
        # gen_mel(model, text_fs2, name, args.duration_control, args.pitch_control, args.energy_control)
        print('DONE', idx)

# print(text_to_sequence('{o2_T1 l e2_T1 sp k o1_T3 t ie2_T3 ng ng uoi3_T2 sp n oi_T3}', ['basic_cleaners']))
