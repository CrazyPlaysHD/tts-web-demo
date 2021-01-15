import os
import numpy as np
import sys
import time
import argparse
import torch
import torch.nn as nn
from scipy.io.wavfile import write
from fastspeech2.model_fs2 import FastSpeech2
from hifi_gan.models import Generator
from hifi_gan.env import AttrDict
import text_fs2
import hparams
from G2p import G2p
from string import punctuation
import re
import json
from pydub import AudioSegment

import IPython
from time import time
import requests
import tacotron2.hparams as hp_tacotron2
from tacotron2.model import Tacotron2
from tacotron2.distributed import apply_gradient_allreduce
from waveglow.denoiser import Denoiser
from numpy import finfo
import text_tacotron

MAX_WAV_VALUE = 32768.0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class T2S:
    def __init__(self, model, vocoder):
        self.model = model
        self.vocoder = vocoder
        self.hparams = hparams
        self.hparams.sampling_rate = 22050
        self.g2p = G2p(hparams.dict_path)


        self.temp_audio = np.zeros(int(0.45 * 22050))
        self.temp_sub_audio = np.zeros(int(0.25 * 22050))


        # load Weight Waveglow
        self.waveglow_path = self.hparams.waveglow_path
        if os.path.exists(self.waveglow_path):
            waveglow = torch.load(self.waveglow_path, map_location=device)['model']
        else:
            waveglow = torch.hub.load(
                'nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
            waveglow = waveglow.remove_weightnorm(waveglow)

        if torch.cuda.is_available():
            waveglow.eval().half()
        else:
            waveglow.eval()

        for m in waveglow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

        for k in waveglow.convinv:
            k.float()

        self.waveglow = waveglow

        # load Weight HifiGan
        h = None
        with open(os.path.join(hparams.hifi_root_path, 'config.json')) as f:
            data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)

        self.generator = Generator(h).to(device)
        state_dict_g = torch.load(os.path.join(hparams.hifi_root_path, 'generator'), map_location=device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        self.denoiser = Denoiser(waveglow)


        # load FastSpeech2
        # self.model_fs2 = self.load_fastspeech2(120000).to(device)
        self.model_fs2 = self.load_fastspeech2().to(device)

        # load Tacotron2
        self.model_tacotron2 = self.load_tacotron2().to(device)

    def load_tacotron2(self):
        hparams = hp_tacotron2.create_hparams()
        hparams.sampling_rate = 22050
        model = Tacotron2(hparams).to(device)

        checkpoint_path = os.path.join(self.hparams.tacotron2_cp_path)

        # model = Tacotron2(hparams).cpu()
        if hparams.fp16_run:
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        if hparams.distributed_run:
            model = apply_gradient_allreduce(model)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
        return model.to(device).eval()

    def normalize(self, text):
        dem = 5
        url = 'http://10.30.132.76:9928/preProcessingApi/get-text'
        rs = None
        while dem > 0:
            dem -= 1
            try:
                with requests.post(url, json={"sentence": text}, timeout=100) as response:
                    if response.status_code != 200:
                        print("FAILURE::{0}".format(url))
                        return {}
                    rs = response.json()
                    break
            except:
                rs = None
                print('\t\tReconnect lan %d' % (20 - dem))
        if rs is None:
            return text
        return rs['new_sentence']


    def preprocess(self, text, use_phone=False):
        text = text.rstrip(punctuation).lower()
        if use_phone:
            phone = self.g2p.g2p(text)
            phone = '{' + '}{'.join(phone.split()) + '}'
            phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
            phone = phone.replace('}{', ' ')
        else:
            # phone = text
            phone = 'z'.join(text.split())
        sequence = np.array(text_fs2.text_to_sequence(phone, hparams.text_cleaners))
        sequence = np.stack([sequence])
        return torch.from_numpy(sequence).long().to(device)
    
    def load_fastspeech2(self):
        checkpoint_path = os.path.join(self.hparams.fastspeech2_cp_path)
        model = nn.DataParallel(FastSpeech2())
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
        model.requires_grad = False
        return model.to(device).eval()

    def save_audio(self, wav, path):
        audio = IPython.display.Audio(wav, rate=hparams.sampling_rate)
        audio = AudioSegment(audio.data, frame_rate=hparams.sampling_rate, sample_width=2, channels=1)
        audio.export(path, format="wav")

    def waveglow_infer(self, mel, sig=1.0, strength=0.01):
        with torch.no_grad():
            if torch.cuda.is_available():
                wav = self.waveglow.infer(mel.half(), sigma=sig)
            else:
                wav = self.waveglow.infer(mel, sigma=sig)
        # print(wav[0].cpu().numpy().shape)
        wav = self.denoiser(wav, strength=strength)[:, 0]
        # print(wav.shape)

        return wav[0].cpu().numpy()

    def hifigan_infer(self, mel, strength=0.01):
        with torch.no_grad():
            if torch.cuda.is_available():
                wav = self.generator(mel)
            else:
                wav = self.generator(mel)
        # print(wav.cpu().numpy().reshape(-1).shape)
        wav = self.denoiser(wav.reshape(1, -1), strength=strength)[:, 0]
        # print(wav.shape)

        return wav.cpu().numpy().reshape(-1)

    def pts(self, para, list_time, dict_input):
        sentence_ls = para.split(".")
        audio = np.zeros(int(0.1 * 22050))
        begin = False

        for idx in range(len(sentence_ls)):
            sen = sentence_ls[idx]
            if sen != '' and sen != ' ':
                sub_stn_ls = re.split(",|;|:", sen)
                begin_sub = False
                audio_sub = np.zeros(int(0.1 * 22050))
                # print(audio_sub.shape)
                for idx_sub in range(len(sub_stn_ls)):
                    sub_stn = sub_stn_ls[idx_sub]
                    if sub_stn != '' and sub_stn != ' ':
                        audio_, times = self.inference_audio(sub_stn, dict_input)
                        # print(audio_.shape)
                        list_time['preprocess'] += times[0]
                        list_time['model_inference'] += times[1]
                        list_time['vocoder_inference'] += times[2]
                        if begin_sub == False:
                            audio_sub = audio_
                            begin_sub = True
                        else:
                            audio_sub = np.concatenate((audio_sub, self.temp_sub_audio), axis=0)
                            audio_sub = np.concatenate((audio_sub, audio_), axis=0)
                if begin == False:
                    audio = audio_sub
                    begin = True
                else:
                    audio = np.concatenate((audio, self.temp_audio), axis=0)
                    audio = np.concatenate((audio, audio_sub), axis=0)
        return audio

    def tts(self, dict_input, filename=None):
        vocoder = dict_input['vocoder']
        model = dict_input['model']
        raw_text = dict_input['text']
        # print(dict_input)

        list_time = {
            'normalize': 0,
            'preprocess': 0,
            'model_inference': 0,
            'vocoder_inference': 0,
        }
        t = time()
        text = self.normalize(raw_text)
        #text = raw_text
        t0 = time()
        list_time['normalize'] = t0 - t
        # print(list_time['normalize'], 'for normalize')
        if filename is None:
            filename = 'samples'
        audio_path = f"{filename}.wav"
        save_path = os.path.join('wavs', audio_path)
        audio = self.pts(text, list_time, dict_input)
        # print("audio saved at: {}".format(save_path))
        self.save_audio(audio, save_path)

        return audio_path, ['Raw Text Input:',
                            '%s' % raw_text,
                            'Normalize text time: %0.3f (s)' % list_time['normalize'],
                            'Preprocessing text time: %0.3f (s)' % list_time['preprocess'],
                            'Model %s inference time: %0.3f (s)' % (model, list_time['model_inference']),
                            'Vocoder %s inference time: %0.3f (s)' % (vocoder, list_time['vocoder_inference']),
                            'Total time: %0.3f (s)' % (time() - t)]


    def inference_audio(self, text, dict_input):
        vocoder = dict_input['vocoder']
        model = dict_input['model']
        # text = dict_input['text']
        d = float(dict_input['d'])
        p = float(dict_input['p'])
        e = float(dict_input['e'])
        sig = float(dict_input['sig'])
        strength = float(dict_input['strength'])
        #print('=============\n\t', text)
        t0 = time()
        list_time = []
        sequence = None
        if model == 'fastspeech2':
            sequence = self.preprocess(text, use_phone=False)
        elif model == 'tacotron2':
            text = (text.strip() + ' .').replace(', .', ' .')
            text = re.sub(' +', ' ', text)
            sequence = np.array(text_tacotron.text_to_sequence(text, ['basic_cleaners']))[None, :]
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence).to(device)).long()

        t1 = time()
        #print(t1 - t0, '(s) for preprocess')
        list_time.append(t1 - t0)

        mel_postnet = None
        if model == 'fastspeech2':
            src_len = torch.from_numpy(np.array([sequence.shape[1]])).to(device)
            mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = self.model_fs2(
                sequence, src_len, d_control=d, p_control=p, e_control=e)
            mel_postnet = mel_postnet.to(device).transpose(1, 2).detach()
        elif model == 'tacotron2':
            mel, mel_postnet, _, alignment = self.model_tacotron2.inference(sequence)

        t2 = time()
        #print(t2 - t1, f'(s) for {model} inference')
        list_time.append(t2 - t1)

        audio = None
        #print(vocoder)
        if vocoder == 'waveglow':
            audio = self.waveglow_infer(mel_postnet, sig=sig, strength=strength)
        elif vocoder == 'hifigan':
            audio = self.hifigan_infer(mel_postnet, strength=strength)
        t3 = time()
        # print(t3 - t2, f'(s) for {vocoder} inference')
        list_time.append(t3 - t2)

        return audio, list_time
        
        

