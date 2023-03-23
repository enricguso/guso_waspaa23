"""!
@brief Training a Sudo-RM-RF model for speech enhancement of the binaural Microson_V1 dataset

"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

from __config__ import API_KEY
from comet_ml import Experiment
import soundfile as sf
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
import numpy as np
import pandas as pd
import improved_cmd_args_parser_v2 as parser
import groupcomm_sudormrf_v2 as sudormrf_gc_v2
import causal_improved_sudormrf_v3 as causal_improved_sudormrf

args = parser.get_args()
hparams = vars(args)
print(hparams)


hparams['separation_task'] == 'enh_noisyreverberant'
hparams['n_sources'] = 1

def SISDR(s, s_hat):
    """Computes the Scale-Invariant SDR as in [1]_.
    References
    ----------
    .. [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?." ICASSP 2019-2019 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
    Parameters:
        s: list of targets of any shape
        s_hat: list of corresponding estimates of any shape
    """
    s = torch.stack(s).view(-1)
    EPS = torch.finfo(s.dtype).eps
    s_hat = torch.stack(s_hat).view(-1)
    a = (torch.dot(s_hat, s) * s) / ((s ** 2).sum() + EPS)
    b = a - s_hat
    return -10*torch.log10(((a*a).sum()) / ((b*b).sum()+EPS))

class microson_v1_dataset(Dataset):
    """Binaural Dataset with RIC on KU100. Microson_V1."""

    def __init__(self, root_dir, split, n):
        """
        Args:
            root_dir (string): Directory with all metadata and data.
            split (string): 'train', 'dev', 'test
            n : number of samples
        """
        self.root_dir = root_dir
        self.split = split
        self.df = pd.read_csv(pjoin(root_dir, 'meta_microson_v1.csv'))
        self.df = self.df[self.df['mls_split'] == split]
        self.n = n

    def __len__(self):
        #return len(self.df)
        return self.n
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wavname = os.path.splitext(row.speech_path)[0]+'.wav'
        anechoic, _ = sf.read(pjoin(pjoin(pjoin(self.root_dir, row.mls_split), 'anechoic'), wavname))
        reverberant, _ = sf.read(pjoin(pjoin(pjoin(self.root_dir, row.mls_split), 'reverberant'), wavname))
        noise, _ = sf.read(pjoin(pjoin(pjoin(self.root_dir, row.mls_split), 'noise'), wavname))
        anechoic = torch.tensor(anechoic.T, dtype=torch.float32)
        reverberant = torch.tensor(reverberant.T, dtype=torch.float32)
        noise = torch.tensor(noise.T, dtype=torch.float32)
        return anechoic, reverberant, noise
    
def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(dim=(1,2), keepdim=True)
    if std is None:
        std = wav_tensor.std(dim=(1,2), keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def log_single_bin(mixtures, targets, estimates, experiment, step):
    for batch_utt in range(mixtures.shape[0]):
        estimate = estimates[batch_utt].detach().cpu().numpy()
        mixture = mixtures[batch_utt].detach().cpu().numpy()
        target = targets[batch_utt].detach().cpu().numpy()

        estimate = estimate / np.abs(estimate).max(-1, keepdims=True)
        mixture = mixture / np.abs(mixture).max(-1, keepdims=True)
        target = target / np.abs(target).max(-1, keepdims=True)

        experiment.log_audio(
                            estimate.T,
                            sample_rate=hparams["fs"],
                            file_name=tag+'_estimate_'+str(batch_utt)+'.wav',
                            copy_to_tmp=True,
                            step=step)
        if step == 0:
            experiment.log_audio(
                                mixture.T,
                                sample_rate=hparams["fs"],
                                file_name=tag+'_mixture_'+str(batch_utt)+'.wav',
                                copy_to_tmp=True,
                                step=step)
            experiment.log_audio(
                                target.T,
                                sample_rate=hparams["fs"],
                                file_name=tag+'_target_'+str(batch_utt)+'.wav',
                                copy_to_tmp=True,
                                step=step)
train_dataset = microson_v1_dataset('/home/ubuntu/Data/microson_v1', 'train', hparams['n_train'])
val_dataset = microson_v1_dataset('/home/ubuntu/Data/microson_v1', 'dev', hparams['n_val'])
test_dataset = microson_v1_dataset('/home/ubuntu/Data/microson_v1', 'test', hparams['n_test'])
train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers = hparams["n_jobs"])
val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers = hparams["n_jobs"])
test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers = hparams["n_jobs"])

if hparams["checkpoints_path"] is not None:
    if hparams["save_checkpoint_every"] <= 0:
        raise ValueError("Expected a value greater than 0 for checkpoint "
                         "storing.")
    if not os.path.exists(hparams["checkpoints_path"]):
        os.makedirs(hparams["checkpoints_path"])

experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

model = causal_improved_sudormrf.CausalSuDORMRF(
        in_audio_channels=2,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['n_sources'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
#experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

#model = torch.nn.DataParallel(model).cuda()
model = model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

for i in range(hparams['n_epochs']):
    train_losses = []
    val_losses = []
    test_losses = []

    print("Training SuDoRM-RF++ GC: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    for anechoic, reverberant, noise in tqdm(train_loader):

        opt.zero_grad()
        # shuffle the noise after the first epoch without changin SNR 
        if hparams['online_mix']:
            if i > 0:
                nrg_noise = torch.sum(noise ** 2, dim=-1, keepdim=True)
                noise = noise[torch.randperm(noise.shape[0])]
                noise *= (torch.sqrt(nrg_noise /
                                    (noise ** 2).sum(-1, keepdims=True)))

        # mix
        mixtures = reverberant + noise
        if hparams['mild_target']:
            reverb = reverberant - anechoic
            targets = anechoic + 0.25 * (reverb + noise)
        else:
            targets = anechoic

        
        if hparams['normalize_online']:
            # normalize mixtures (mean 0 std 1)
            mixtures = normalize_tensor_wav(mixtures)
            # target normalization also seems to help (0.5dB in val set)
            targets = normalize_tensor_wav(targets) #this overrides target SNR augmentation

        mixtures = mixtures.cuda()
        targets = targets.cuda()

        estimates = model(mixtures)
        

        l = torch.clamp(SISDR([targets], [estimates]), min=-50, max=+50)
        l.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])
        opt.step()
        train_losses.append(l.detach())

    # lr_scheduler.step(res_dic['val_SISDRi']['mean'])
    if hparams['patience'] > 0:
        if i % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (i // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    model.eval()
    with torch.no_grad():
        for anechoic, reverberant, noise in tqdm(val_loader,
                            desc='Validation...'):
            mixtures = reverberant + noise
            if hparams['normalize_online']:
                mixtures = normalize_tensor_wav(mixtures)
            if hparams['mild_target']:
                reverb = reverberant - anechoic
                targets = anechoic + 0.25 * (reverb + noise)
            else:
                targets = anechoic
            mixtures = mixtures.cuda()
            targets = targets.cuda()
            estimates = model(mixtures)

            l = SISDR([targets], [estimates])
            val_losses.append(l.detach())

        for anechoic, reverberant, noise in tqdm(test_loader,
                            desc='Testing...'):
            mixtures = reverberant + noise
            if hparams['normalize_online']:
                mixtures = normalize_tensor_wav(mixtures)
            if hparams['mild_target']:
                reverb = reverberant - anechoic
                targets = anechoic + 0.25 * (reverb + noise)
            else:
                targets = anechoic
            mixtures = mixtures.cuda()
            targets = targets.cuda()
            estimates = model(mixtures)

            #for loss_name, loss_func in all_losses[val_set].items():
            l = SISDR([targets], [estimates])
            test_losses.append(l.detach())

    experiment.log_metric(name = "tr_sisdr_mean", value= (-torch.stack(train_losses).mean()).cpu().detach().numpy(), epoch=i)
    experiment.log_metric(name = "tr_sisdr_std", value= (torch.stack(train_losses).std()).cpu().detach().numpy(), epoch=i)
    experiment.log_metric(name = "val_sisdr_mean", value= (-torch.stack(val_losses).mean()).cpu().detach().numpy(), epoch=i)
    experiment.log_metric(name = "val_sisdr_std", value= (torch.stack(val_losses).std()).cpu().detach().numpy(), epoch=i)
    experiment.log_metric(name = "tt_sisdr_mean", value= (-torch.stack(test_losses).mean()).cpu().detach().numpy(), epoch=i)
    experiment.log_metric(name = "tt_sisdr_std", value= (torch.stack(test_losses).std()).cpu().detach().numpy(), epoch=i)
    log_single_bin(mixtures, targets, estimates, experiment, i)

    if hparams["save_checkpoint_every"] > 0:
        if i % hparams["save_checkpoint_every"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(hparams["checkpoints_path"],
                             f"gc_sudo_epoch_{i}"),
            )
