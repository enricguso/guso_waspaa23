import os
import sys
import torchaudio 
import torch
from tqdm import tqdm
import numpy as np
import groupcomm_sudormrf_v2 as sudormrf_gc_v2
import causal_improved_sudormrf_v3 as causal_improved_sudormrf
import argparse

parser = argparse.ArgumentParser(
        description='Experiment Argument Parser')
parser.add_argument("--model_path", type=str,
                help="""Folder containing the checkpoint of the model.""",
                default=None)
parser.add_argument("--input", type=str,
                help="""Directory from where to load wavs.""",
                default=None)
parser.add_argument("--output", type=str,
                help="""Directory where to save processed wavs.""",
                default=None)
args = parser.parse_args()

if os.path.basename(args.model_path).split('_')[-1] == 'causal':
    model = causal_improved_sudormrf.CausalSuDORMRF(
            in_audio_channels=2,
            out_channels=512,
            in_channels=256,
            num_blocks=16,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=1)
else:
    model = sudormrf_gc_v2.GroupCommSudoRmRf(
            in_audio_channels=2,
            out_channels=512,
            in_channels=256,
            num_blocks=16,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=1)

checkpoint_name = os.listdir(args.model_path)
checkpoint_name.sort()
model_path = os.path.join(args.model_path, checkpoint_name[0])
model.load_state_dict(torch.load(model_path))
model.eval()

infiles = os.listdir(args.input)
infiles = [k for k in infiles if '.wav'  in k]
infiles = [k for k in infiles if '.reapeaks' not in k]
infiles.sort()

for f in infiles:
    mixture, fs = torchaudio.load(os.path.join(args.input, f))
    ini_nrg = torch.sum(mixture ** 2)
    mixture = (mixture - torch.mean(mixture)) / torch.std(mixture)
    denoised = model(mixture.unsqueeze(0)).detach()
    print(denoised.shape)