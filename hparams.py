dataset = "vlsp2020"
data_path = '/content/vlsp2020/'
name_task = 'FastSpeech2'  # 'FastSpeech2'
waveglow_path = '/content/drive/MyDrive/voice_data/waveglow_78000'
# hifigan_path = '/u01/os_chatbot/tts_fastspeech-master/hifigan_78000'

# hifi_root_path = '/workspace/tts_models/hifigan_pretrained/pretrained/LJ_V1/'
hifi_root_path = '/workspace/tts_models/hifigan_trained/'

tacotron2_cp_path = '/content/drive/MyDrive/voice_data/checkpoint_42000'
fastspeech2_cp_path = '/content/drive/MyDrive/voice_data/FS2_MFA_20200116_300k_char/ckpt/vlsp2020/checkpoint_880000.pth.tar'
checkpoint_path = '/content/tts-web-demo/'

dict_path = '/content/tts-web-demo/syllable_g2p.txt'
# Text
# text_cleaners = ['basic_cleaners']
text_cleaners = []

# Audio and mel
### for VLSP2020 ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

# Quantization for F0 and energy
### for VLSP2020 ###
f0_min = 71.0
f0_max = 786.6
energy_min = 0.0
energy_max = 321.4


# Vocoder
vocoder = 'waveglow'  # 'waveglow' or 'melgan'

# Log-scaled duration
log_offset = 1.

# Save, log and synthesis
save_step = 20000
synth_step = 10000
eval_step = 10000
eval_size = 256
log_step = 1000
clear_Time = 20

n_bins = 256

batch_size = 16
epochs = 1000
batch_expand_size = 16

max_seq_len = 1000
