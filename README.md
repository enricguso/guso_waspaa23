# microson_v1
Generate a binaural speech enhancement dataset with speech from Multilingual LibriSpeech Spanish + WHAM! binaural noises.

### Install dependencies:
For installation we recommend to create a virtual environment as follows:
```
python -m venv <venv_name>
```
And install dependencies:
```
pip install numpy scipy mat73 jupyter soundfile pyrubberband matplotlib pandas ipykernel tqdm
```
Finally add the virtualenv kernel to jupyter:
```
python -m ipykernel install --user --name <venv_name>
```
### Listen to the different decoders (Optional)
Open jupyter notebook while choosing your environment.
Run ```debug_decoders.ipynb``` and choose a decoder by changing the decoder path between:
* decoders_ord10/KU100_ALFE_Window_sinEQ_bimag.mat: 50-point HRIRs 10th-order Ambisonics to Binaural decoder from the KU100 dummy.
* decoders_ord10/RIC_Front_Omni_ALFE_Window_sinEQ_bimag.mat: 50-point HRIRs 10th-order Ambisonics to Binaural decoder from the KU100 dummy wearing a hearing aids device.

### Generate metadata for the dataset (Optional):
Open jupyter notebook while choosing your environment.
Run ```microson_v1_dataset_design.ipynb``` with your WHAM! and Multilingual LibriSpeech Spanish dataset paths.

Generates ```meta_microson_v1.csv```.

### Generate the audio:
Run ```generate_microsonv1.py``` specifying again the dataset input and output paths, the path to the CSV metadata file, and the desired decoder.