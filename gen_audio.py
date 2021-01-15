from text2speech import T2S
import os

model = 'tacotron2'
vocoder = 'hifigan'
t2s = T2S(model, vocoder)

path = '/workspace/cuongnm55/tts_data/metadata.csv'
list_time = {
            'normalize': 0,
            'preprocess': 0,
            'model_inference': 0,
            'vocoder_inference': 0,
        }

dict_input = {}

dict_input['vocoder'] = vocoder
dict_input['model'] = model
dict_input['text'] = ''
dict_input['d'] = 1.0
dict_input['p'] = 1.0
dict_input['e'] = 1.0
dict_input['sig'] = 1.0
dict_input['strength'] = 0.22

with open(path, 'r', encoding='utf-8') as rf:
    lines = rf.read().split('\n')
    dem = 0
    for i, line in enumerate(lines):
        if '|' in line:
            filename = line.split('|')[0]
            text = line.split('|')[-1]
            audio = t2s.pts(text, list_time, dict_input)
            audio_path = f"{filename}.wav"
            save_path = os.path.join('/workspace/cuongnm55/tts_data/wavs', audio_path)
            t2s.save_audio(audio, save_path)
            with open(os.path.join('/workspace/cuongnm55/tts_data/wavs/', filename + '.txt'), 'w', encoding='utf8') as wf:
                wf.write(text.strip())
            dem += 1
            if dem % 100 == 0:
                print('DONE', dem, '/', len(lines))



