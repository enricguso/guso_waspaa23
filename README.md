# ha_enhancement_experiment
We compare 4 different binaural renderers:
* KU100woHAinear: Eurecat measurement of the KU100 in our studio, without a hearing aids device.
* KU100wHAinear: Eurecat measurement of the KU100 in our studio, with a hearing aids device with all features disabled (only amplifying) and recording the in-ear mic.
* BKwHAinear: Oldenburg measurement of the B&K dummy, wearing a hearing aids (turned-off) but recording the in-ear mics.
* BKwHAHAmic: Oldenburg measurement of the B&K dummy, this time recording the Front microphones of the HA device.

### Install dependencies:
For installation we recommend to create a virtual environment as follows:
```
python -m venv <venv_name>
```
And install dependencies:
```
pip install -r requirements.txt
```

### Listen to the different decoders
Open jupyter notebook while choosing your environment.
Run ```compare_decoders.ipynb```
