import numpy as np
from IPython.display import Audio
import scipy.signal as sig
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal



# ----------- FUNCTION DEFINITIONS: -----------
def bell(fc, fs, gain, Q):
    wc = 2 * np.pi * fc / fs
    c = 1.0 / np.tan(wc / 2.0)
    phi = c*c
    Knum = c / Q
    Kdenom = Knum

    if (gain > 1.0):
        Knum *= gain
    elif (gain < 1.0):
        Kdenom /= gain

    a0 = phi + Kdenom + 1.0

    b = [(phi + Knum + 1.0) / a0, 2.0 *
         (1.0 - phi) / a0, (phi - Knum + 1.0) / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - Kdenom + 1.0) / a0]

    return np.asarray(b), np.asarray(a)

def power(signal):
    return np.mean(signal**2)

def extend_noise(noise, max_length, fs):
    """ Concatenate noise using hanning window"""
    noise_ex = noise
    window = np.hanning(fs + 1)
    # Increasing window
    i_w = window[:len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2::-1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   np.multiply(
                                       noise_ex[len(noise_ex) - len(d_w):],
                                       d_w) + np.multiply(
                                       noise[:len(i_w)], i_w),
                                   noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex

def crop_echogram(anechoic_echogram):
    nSrc = anechoic_echogram.shape[0]
    nRec = anechoic_echogram.shape[1]
    nBands = anechoic_echogram.shape[2]
    # Returns the "anechoic" version of an echogram
    # Should keep the receiver directivy
    for src in range(nSrc):
        for rec in range(nRec):
            for band in range(nBands):
                anechoic_echogram[src, rec, band].time = anechoic_echogram[src, rec, band].time[:2]
                anechoic_echogram[src, rec, band].coords = anechoic_echogram[src, rec, band].coords[:2, :]
                anechoic_echogram[src, rec, band].value = anechoic_echogram[src, rec, band].value[:2,:]
                anechoic_echogram[src, rec, band].order = anechoic_echogram[src, rec, band].order[:2,:]
    return anechoic_echogram

# def align_signals(s1,s2):
#     corr = sig.correlate(s1[:48000,1],s2[:48000,1],mode='full')
#     plt.figure()
#     plt.xcorr(1[:48000,1],s2[:48000,1])
#     plt.show()
#     # shift = np.argmax(np.abs(corr))
#     # print(shift)
#     # plt.figure()
#     # plt.plot(s1)
#     # plt.show()
#     # s1=s1[shift:,:]
#     # plt.figure()
#     # plt.plot(s1)
#     # plt.show()
#     s1=np.concatenate((s1,np.zeros((shift,2))),axis=0)
#     return s1,s2


def place_on_circle(head_pos,r,angle_deg):
# place a source around the reference point (like head)
    angle_rad = (90-angle_deg) * (np.pi / 180)
    x_coord=head_pos[0]+r*np.sin(angle_rad)
    y_coord=head_pos[1]+r*np.cos(angle_rad)
    src_pos=np.array([x_coord, y_coord, head_pos[2]]) 
    return [src_pos]

def place_on_circle_in_room(head_pos,r,angle_deg, room):
# place a source around the reference point (like head)
# reducing distance if needed
    src_pos = place_on_circle(head_pos, r, angle_deg)[0]
    #x_coord=head_pos[0]+r*np.sin(angle_rad)
    #y_coord=head_pos[1]+r*np.cos(angle_rad)
    #check if x_coord and y_coord are outside the room:
    #src_pos=np.array([x_coord, y_coord, head_pos[2]]) 
    src_pos[src_pos < 0.2] = 0.2
    while np.any(src_pos > room - 0.2):
        r*=0.9
        src_pos = place_on_circle(head_pos, r, angle_deg)[0]
        src_pos[src_pos < 0.2] = 0.2
    return src_pos

def head_2_ku_ears(head_pos,head_orient):
# based on head pos and orientation, compute coordinates of ears
    ear_distance_ku100=0.0875
    theta = (head_orient[0]) * np.pi / 180
    R_ear = [head_pos[0] - ear_distance_ku100 * np.sin(theta),
              head_pos[1] + ear_distance_ku100 * np.cos(theta), 
              head_pos[2]]
    L_ear = [head_pos[0] + ear_distance_ku100 * np.sin(theta),
              head_pos[1] - ear_distance_ku100 * np.cos(theta), 
              head_pos[2]]
    return [L_ear,R_ear]

'''
def head_2_ku_ears(head_pos,head_orient):
# based on head pos and orientation, compute coordinates of ears
    ear_distance_ku100=0.0875
    theta = (90-head_orient[0]) * np.pi / 180
    R_ear = [head_pos[0] - ear_distance_ku100 * np.cos(theta),
              head_pos[1] + ear_distance_ku100 * np.sin(theta), 
              head_pos[2]]
    L_ear = [head_pos[0] + ear_distance_ku100 * np.cos(theta),
              head_pos[1] - ear_distance_ku100 * np.sin(theta), 
              head_pos[2]]
    return [L_ear,R_ear]
'''

def add_signals(a,b):
# add values of two arrays of different lengths
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c


def plot_scene(room_dims,head_pos,head_orient,l_mic_pos,l_src_pos,perspective="xy"):
#   function to plot the designed scene
#   room_dims - dimensions of the room [x,y,z]
#   head_pos - head position [x,y,z]
#   head_orient - [az,el]
#   l_src_pos - list of source positions [[x,y,z],...,[x,y,z]]
#   perspective - which two dimensions to show 
    if perspective=="xy":
        dim1=1
        dim2=0
    elif perspective=="yz":
        dim1=2
        dim2=1
    elif perspective=="xz":
        dim1=2
        dim2=0
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim((0,room_dims[dim1]))
    plt.ylim((0,room_dims[dim2]))
    plt.axvline(head_pos[dim1], color='y') # horizontal lines
    plt.axhline(head_pos[dim2], color='y') # vertical lines
    plt.grid(True)
    # plot sources and receivers
    plt.plot(head_pos[dim1],head_pos[dim2], "o", ms=10, mew=2, color="black")
    # plot ears
    plt.plot(l_mic_pos[0][dim1],l_mic_pos[0][dim2], "o", ms=3, mew=2, color="blue")# left ear in blue
    plt.plot(l_mic_pos[1][dim1],l_mic_pos[1][dim2], "o", ms=3, mew=2, color="red")# right ear in red

    for i,src_pos in enumerate(l_src_pos):
        plt.plot(src_pos[dim1],src_pos[dim2], "o", ms=10, mew=2, color="red")
        plt.annotate(str(i), (src_pos[dim1],src_pos[dim2]))
    # plot head orientation if looking from above 
    if perspective=="xy":
        plt.plot(head_pos[dim1],head_pos[dim2], marker=(1, 1, -head_orient[0]), ms=20, mew=2,color="black")

    ax.set_aspect('equal', adjustable='box')


def set_level(sig_in,L_des):
# set FS level of the signal
    sig_zeromean=np.subtract(sig_in,np.mean(sig_in,axis=0))
    sig_norm_en=sig_zeromean/np.std(sig_zeromean.reshape(-1))
    sig_out =sig_norm_en*np.power(10,L_des/20)
    print(20*np.log10(np.sqrt(np.mean(np.power(sig_out,2)))))
    return sig_out

def generate_scenes(sources_sigs,levels,mic_rirs,decoder):
# generate binaural mixture signal based on generated irs, binaural decoder and source signals
    sig_L_mix=np.zeros(100)
    sig_R_mix=np.zeros(100)
    for i, source_sig in enumerate(sources_sigs):
        #mic_rirs[:, :, ear, source]
        filter_L=sig.fftconvolve(np.squeeze(mic_rirs[:,:,0, i]).T, decoder[:,:,0].T, 'full', 1).sum(0)
        filter_R=sig.fftconvolve(np.squeeze(mic_rirs[:,:,1, i]).T, decoder[:,:,1].T, 'full', 1).sum(0)
        # set level for current source BEFORE SPATIALIZING:
        source_sig=set_level(source_sig.T,levels[i])
        # spatialize:
        sig_L=sig.fftconvolve(source_sig, filter_L, 'full')
        sig_R=sig.fftconvolve(source_sig, filter_R, 'full')
        # # set level for current source AFTER SPATIALIZING:
        # sig_LR_leveled=set_level(np.array((sig_L,sig_R)).T,levels[i])
        sig_LR_leveled=np.array((sig_L,sig_R)).T
        # add generated source signal to the mixture using a function that takes variable signal lenghts
        sig_L_mix=add_signals(sig_L_mix,sig_LR_leveled[:,0])# left channel
        sig_R_mix=add_signals(sig_R_mix,sig_LR_leveled[:,1])# right channel
        # put left and right signal into one array
        mix=np.array((sig_L_mix,sig_R_mix))

    return mix


def synch_sigs(sig1,sig2):
    sig1_out=np.zeros(sig1.shape)
    sig2_out=np.zeros(sig2.shape)
    corr = signal.correlate(sig1[:,0], sig2[:,0], 'full')
    lag = signal.correlation_lags(len(sig1[:,0]), len(sig2[:,0]), mode='full')[np.argmax(corr)]
    if lag > 0:
        sig2=sig2[0:-lag, :]
        sig1=sig1[lag:, :]
    elif lag < 0:
        sig2=sig2[-lag:, :]
        sig1=sig1[0:lag, :]

    sig1_out[:sig1.shape[0],:]=sig1
    sig2_out[:sig2.shape[0],:]=sig2
    return sig1_out,sig2_out

def generate_sig_in_SH(source_sig,sh_mic_rir):
# function to generate 1 source signal in spherical harmonics
# (representing the source in a room).
# --- Input:---
# source_sig - mono audio signal (1, L_sig)
# amb_mic_rir - RIRs in SH computed using masp (121, L_ir)
# --- Output:---
# sh_out - signal in spherical harmonics (121, L_conv), where L_conv= L_sig+L_ir-1 

    sig_broadcasted=np.tile(source_sig,[121,1])
    sh_out=sig.oaconvolve(sig_broadcasted, sh_mic_rir, 'full',-1)
    return sh_out

def sh_1mic_to_binaural(sh_in,decoder):
# function to generate binaural signal using binaural decoder
# which takes as input source signal in sh for 1 microphone position
# --- Input:---
# sh_in - source signal in sh for 1 mic position (121, L_conv), where L_conv= L_sig+L_ir-1 
# decoder - ambisonics binaural decoder (L_dec,121,2)
# --- Output:---
# bin_out - binaural signal after convolution with a decoder

    assert(sh_in.ndim == 2)
    assert(decoder.ndim == 3)
    assert(sh_in.shape[0] == decoder.shape[1])
    out_l = np.sum(signal.oaconvolve(sh_in, np.squeeze(decoder[:, :, 0].T),
                                     axes=-1), axis=0)
    out_r = np.sum(signal.oaconvolve(sh_in, np.squeeze(decoder[:, :, 1].T),
                                     axes=-1), axis=0)
    return np.vstack((out_l, out_r))

def sh_2mics_to_binaural(sh_in_L,sh_in_R,decoder):
# function to generate binaural signal using binaural decoder
# which takes as input source signals in sh for 2 microphone positions
# corresponding to ear positions
# --- Input:---
# sh_in_L - source signal in sh for left mic position (121, L_conv), where L_conv= L_sig+L_ir-1 
# sh_in_R - source signal in sh for right mic position (121, L_conv), where L_conv= L_sig+L_ir-1 
# decoder - ambisonics binaural decoder (L_dec,121,2)
# --- Output:---
# bin_out - binaural signal after convolution with a decoder

    assert(sh_in_R.ndim == sh_in_L.ndim)
    assert(sh_in_R.ndim == 2)
    assert(sh_in_R.ndim == 2)
    assert(decoder.ndim == 3)
    assert(sh_in_R.shape[0] == decoder.shape[1])
    out_l = np.sum(signal.oaconvolve(sh_in_L, np.squeeze(decoder[:, :,0].T),
                                     axes=-1), axis=0)
    out_r = np.sum(signal.oaconvolve(sh_in_R, np.squeeze(decoder[:, :, 1].T),
                                     axes=-1), axis=0)
    return np.vstack((out_l, out_r))


def decode_noise(MOA_noise,level,decoder):
    MOA_noise=set_level(MOA_noise.T,level[0])
    # convolve signal in ambisonics domain with a decoder
    # decoder[:,:,0] - for left ear
    # decoder[:,:,1] - for the right ear
    noise_L=sig.fftconvolve(MOA_noise, decoder[:,:,0].T, 'full', 1).sum(0)
    noise_R=sig.fftconvolve(MOA_noise, decoder[:,:,1].T, 'full', 1).sum(0)
    noise=np.array((noise_L,noise_R))
    return noise


    


