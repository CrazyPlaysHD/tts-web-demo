import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os

import time

from dataset.dataset_fs2 import Dataset
from fastspeech2.model_fs2 import FastSpeech2
from fastspeech2.evaluate_fs2 import evaluate
from fastspeech2.loss_fs2 import FastSpeech2Loss
from fastspeech2.optimizer_fs2 import ScheduledOptim
from fastspeech2 import hp_fs2 as hp2

import hparams as hp

import utils
# import audio as Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args, device):
    torch.manual_seed(0)

    hp.checkpoint_path = os.path.join(hp.root_path, args.name_task, "ckpt", hp.dataset)
    hp.synth_path = os.path.join(hp.root_path, args.name_task, "synth", hp.dataset)
    hp.eval_path = os.path.join(hp.root_path, args.name_task, "eval", hp.dataset)
    hp.log_path = os.path.join(hp.root_path, args.name_task, "log", hp.dataset)
    hp.test_path = os.path.join(hp.root_path, args.name_task, 'results')

    list_unuse = []

    # Get device


    # Get dataset
    dataset = Dataset("train.txt", list_unuse)
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # Define model
    model = nn.DataParallel(FastSpeech2()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), betas=hp2.betas, eps=hp2.eps, weight_decay=hp2.weight_decay)
    scheduled_optim = ScheduledOptim(
        optimizer, hp2.decoder_hidden, hp2.n_warm_up_step, args.restore_step)
    # Optimizer and loss)



    Loss = FastSpeech2Loss().to(device)
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(args.restore_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        if 'LJSpeech' in checkpoint_path:
            pretrained_dict = checkpoint['model']
            model_dict = model.state_dict()
            for k, v in pretrained_dict.items():
                print(k)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except Exception as e:
        print(e)
        print("\n---Start New Training---\n")
        checkpoint_path = os.path.join(hp.checkpoint_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    for i in args.list_freeze:
        module = getattr(model, i)
        module.weight.requires_grad = False
        module.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=hp2.betas,
        eps=hp2.eps,
        weight_decay=hp2.weight_decay)
    scheduled_optim = ScheduledOptim(
        optimizer, hp2.decoder_hidden, hp2.n_warm_up_step, args.restore_step)

    # Load vocoder
    if hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))

    train_logger = SummaryWriter(os.path.join(log_path, 'train'))

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    current_step = args.restore_step
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = current_step + (hp.epochs - epoch) * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + \
                    epoch*len(loader)*hp.batch_size + 1

                scheduled_optim.zero_grad()

                # Get Data
                text = torch.from_numpy(
                    data_of_batch["text_fs2"]).long().to(device)
                mel_target = torch.from_numpy(
                    data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(
                    data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(
                    data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(
                    data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(
                    data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")
                with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
                    f_f_loss.write(str(f_l)+"\n")
                with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
                    f_e_loss.write(str(e_l)+"\n")

                # Backward
                total_loss = total_loss / hp2.acc_steps
                total_loss.backward()
                if current_step % hp2.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp2.grad_clip_thresh)

                # Update weights
                if args.frozen_learning_rate:
                    scheduled_optim.step_and_update_lr_frozen(
                        args.learning_rate_frozen)
                else:
                    scheduled_optim.step_and_update_lr()


                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str0 = 'Training:'

                    str1 = "\tEpoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "\tTotal Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "\tCurrent Learning Rate is {:.6f}.".format(
                        scheduled_optim.get_learning_rate())
                    str4 = "\tTime Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str0)
                    print(str1)
                    print(str2)
                    print(str3)
                    print(str4)

                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    train_logger.add_scalar(
                        'Loss/total_loss', t_l, current_step)
                    train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    train_logger.add_scalar(
                        'Loss/mel_postnet_loss', m_p_l, current_step)
                    train_logger.add_scalar(
                        'Loss/duration_loss', d_l, current_step)
                    train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                    train_logger.add_scalar(
                        'Loss/energy_loss', e_l, current_step)

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))

                if current_step % hp.synth_step == 0:
                    length = mel_len[0].item()
                    mel_target_torch = mel_target[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    # mel_target = mel_target[0, :length].detach(
                    # ).cpu().transpose(0, 1)
                    mel_torch = mel_output[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    # mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                    mel_postnet_torch = mel_postnet_output[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    # mel_postnet = mel_postnet_output[0, :length].detach(
                    # ).cpu().transpose(0, 1)
                    # Audio.tools.inv_mel_spec(mel, os.path.join(
                    #     synth_path, "step_{}_griffin_lim.wav".format(current_step)))
                    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
                    #     synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))

                    if hp.vocoder == 'waveglow':
                        utils.waveglow_infer(mel_torch, waveglow, os.path.join(
                            hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(
                            hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_target_torch, waveglow, os.path.join(
                            hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))

                    # f0 = f0[0, :length].detach().cpu().numpy()
                    # energy = energy[0, :length].detach().cpu().numpy()
                    # f0_output = f0_output[0, :length].detach().cpu().numpy()
                    # energy_output = energy_output[0,
                    #                               :length].detach().cpu().numpy()
                    #
                    # utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output), (mel_target.numpy(), f0, energy)],
                    #                 ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, m_l, m_p_l = evaluate(
                            model, current_step)
                        t_l = d_l + f_l + e_l + m_l + m_p_l
                        str0 = 'Validating'
                        str1 = "\tEpoch [{}/{}], Step [{}/{}]:".format(
                            epoch + 1, hp.epochs, current_step, total_step)
                        str2 = "\tTotal Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                            t_l, m_l, m_p_l, d_l, f_l, e_l)
                        print(str0)
                        print(str1)
                        print(str2)
                    model.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args, device)
