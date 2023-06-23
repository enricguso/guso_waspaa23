# microson_v1
Code to replicate our WASPAA23 submission: AN OBJECTIVE EVALUATION OF HEARING AIDS AND DNN-BASED BINAURAL SPEECH ENHANCEMENT IN COMPLEX ACOUSTIC SCENES, where we benchmark traditionald DNN beamforming techniques against DNN-based enhancement.
>Enric Gusó enric.guso@eurecat.org

>Joanna Luberazdka joanna.luberadzka@eurecat.org

Code to replicate our WASPAA23 submission

* Generate a Hearing Aid binaural speech enhancement (denoising + dereverberation) dataset with speech from Multilingual LibriSpeech Spanish + WHAM! binaural noises.

* Train and evaluate a Sudo RM-RF real-time oriented enhancement DNN on that dataset.

* Generate a small test set of 10-th order Ambisonics situations that we use for recording different Hearing Aids in bypass and using the HA enhancement.

* Process the recordings in bypass with a Causal DNN (DNN-C) and a non-causal upper baseline (DNN).

* Evaluate in terms of SISDR, HASPI, HASQI and MBSTOI

<img src="figures/results.png" alt="isolated" width="440"/>

## Dependencies:

Then install dependencies. For the dataset creation:
```
pip install numpy scipy mat73 jupyter soundfile pyrubberband matplotlib pandas ipykernel tqdm 
```
And dependencies for the DNN training and evaluation:
```
pip install comet_ml torch pyclarity seaborn
```

Alternative, install specific versions from file with:
```
pip install -r requirements.txt
```

## DNN Dataset Generation

### Generate metadata for the dataset

Generate a metadata dataframe for augmenting WHAM! to match the size of the speech dataset. 

<img src="figures/table.png" alt="isolated" width="340"/>

Open jupyter notebook while choosing your environment.
Run ```microson_v1_dataset_design.ipynb``` with your WHAM! and Multilingual LibriSpeech Spanish (MLSS) dataset paths.
### Listen to the different decoders (Optional)
Run ```debug_notebooks/debug_decoders.ipynb``` and choose a decoder by changing the decoder path between:
* decoders_ord10/KU100_ALFE_Window_sinEQ_bimag.mat: 50-point HRIRs 10th-order Ambisonics to Binaural decoder from the KU100 dummy.
* decoders_ord10/RIC_Front_Omni_ALFE_Window_sinEQ_bimag.mat: 50-point HRIRs 10th-order Ambisonics to Binaural decoder from the KU100 dummy wearing a hearing aids device.

>Generates ```meta_microson_v1.csv```.
### Generate the audio
Edit paths and parameters (if needed) at the end of ```generate_microsonv1.py``` script and then run it.
We obtain the following wav files in ```output_dir```:
* ```ane_ir```: the anechoic impulse response in binaural
* ```anechoic```: the anechoic speech signal in binaural
* ```reverberant```: the reverberant speech signal in binaural
* ```ir```: the reverberant impulse response in binaural
* ```mono_ir```: the reverberant impulse response in mono
* ```noise```: a corresponding augmented chunk from WHAM!
## Model training

Go to ```sudo_rm_rf``` folder.
Configure your CometML API Key in ```__config__.py``` file. Also adjust the output (checkpoints) path for each model:
* ```m1_alldata_normal.sh``` for a non-causal model where ```target=anechoic```
* ```m3_alldata_mild.sh``` non-causal where ```target = anechoic + 0.25*(reverb + noise)```
* ```m4_alldata_normal_causal.sh``` for a causal model where ```target=anechoic```
* ```m5_alldata_mild_causal``` causal where ```target = anechoic + 0.25*(reverb + noise)```

Then run (the biggest model took about 20days in a DATURA V100 instance) at each server with ```sh <script_name>.sh```

Once training is complete, pick the best epoch (by visual inspection) from the checkpoints folder (should always be the last one in this case.)

## Test Set of Complex Situations

Download ```02_Office_MOA_31ch.wav```, ```07_Cafe_1_MOA_31ch.wav'``` and ```09_Dinner_party_MOA_31ch.wav'``` from the Ambisonics Recordings of Typical environments (ARTE) Database and place in a directory:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2261633.svg)](https://doi.org/10.5281/zenodo.2261633)

Run ```listening_test_scenes.ipynb``` twice. Firstly as it is, and then changing the ```tag``` variable from ```normal```to ```inverse``` at the beggining of the notebook.

Then decode the resulting ambisonic signals to speaker signals using the tool that suits your particular speaker setup (e.g. AllRAD) and normalize so that all utterances in the speaker signal test set have the same energy overall.

Calibrate the speaker setup to 70dB and record the different HA in bypass (without beamforming and other traditional enhancement methods) and enabling them (enabled).

Crop these recordings with ```recordings_crop.ipynb``` and process with ```recordings_process.ipynb``` while adjusting the paths if necessary.

Finally run ```recordings_analysis_bintarget.ipynb``` for computing the metrics and generating the plots. Results are stored in ```results.csv```.