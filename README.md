# CCSBase2

A Collision Cross Section prediction model using a XGBoostRegressor model.

## Dependencies

### 1. Create a Conda environment (Python 3.10)

```bash
conda create -n ccsbase2 python=3.10 -y
conda activate ccsbase2
```

### 2. Install
```bash
pip install numpy pandas scikit-learn rdkit xgboost joblib requests matplotlib
```

## Building Database

### Build Main Database
CCSbase2.0 aggregates across the 5 different datasets listed below. AllCCS was obtained by downloading chemicals manually using their online database.


#### CCSBase: https://ccsbase.net/

#### PNNL: https://pnnl-comp-mass-spec.github.io/MetabolomicsCCS/

#### ALLCCS: http://allccs.zhulab.cn/database/browser

#### METLIN: https://www.dropbox.com/scl/fi/9xctm5ub834muw1qrvd5b/CCS-Publication-V3.zip?e=2&file_subpath=%2FCCS-Publication-V3&rlkey=zi9xaua4zzgpiiaznexabpg7i&dl=0

#### ACS Publication: https://pubs.acs.org/doi/10.1021/acs.jafc.2c00724


The full dataset used for our models, including prebuilt 3D conformers, is available for download. In addition, model weights are provided for models trained under different dataset configurations.

### Prebuilt Database
- [CCSMLDatabase](https://drive.google.com/file/d/1NQy1ZcuRwRlZv2scIgqvrvsFDLFhEewx/view?usp=drive_link)


### Model Weights
- [Trained with Full Dataset](https://drive.google.com/file/d/1gIwU1uCSAY__sjbu3eCLDnJ8KqZ6B06E/view?usp=sharing)

- [Trained without METLIN](https://drive.google.com/file/d/1OTsnFYngu1EtKoHOkks6lS_HHEgYnh1B/view?usp=sharing)

- [Trained without METLIN and with `subclass_frequency_threshold ≥ 40`](https://drive.google.com/file/d/1Ohktmduvr6eid-pBRUIUg7D5Fs-NsRra/view?usp=sharing)



Alternatively, you can build the database from scratch. Run the code below in a separate file. Please note that this entire process takes a very long time as thousands of API calls are being made.

We advise you to run this in Google Colab due to PubChem rate limiting IP addresses which causes APIs to retrieve a "ServerBusy" error. Because of this, you may need to run ``ccsml.find_smiles()`` multiple times while commenting out the rest of the method calls. ClassyFire also does rate limiting, so you will need to run ``ccsml.find_inchikey()`` multiple times while commenting out all other method calls.

You will also need to run ``ccsml.find_inchikey()`` at the end which calls Pubchem API to retrieve InChiKey when given SMILES string. This is to obtain InChiKeys for CCSbase datapoints since CCSbase stores SMILES, but not InChiKeys. 

In total, this process takes 1-2 days depending on the number of rate limit errors.

```python
from data import CCSDataIntegration

ccsml = CCSDataIntegration("CCSMLDatabase.db")
ccsml.add_acs()
ccsml.add_ccsbase()
ccsml.add_allccs()
ccsml.add_metlin()
ccsml.add_pnnl()
ccsml.find_smiles()
ccsml.find_inchikey()
ccsml.find_classes()
ccsml.clean()
```


### Classify Unknown Subclasses

After building the database, you'll find datapoints that have a valid SMILES and InChiKey, however ClassyFire could not identify a valid SMILES string. In these cases, we have built a XGBoostClassifier that can classify the subclasses of these compounds, which increases the amount of data the model sees. All subclasses in our dataset that have less than 30 points are not included during training and act as a proxy out of distribution (OOD) dataset. This process takes around 45 minutes to train on a M4 Pro Macbook Pro.

```python
from classifier import SubclassClassifier

classifier = SubclassClassifier("CCSMLDatabase.db", min_class_count=30, seed=26)
classifier.fit()
classifier.predict()
```


### Build 3D Conformers (Optional)

Abalations have been ran using 3D conformers, however all models we have trained using 3D descriptors perform suboptimally from the current model. This is mostly due to ambiguity with determining the correct conformer that a chemical compound took on while in the mass spectrometry instrument. However, if you would like to try experiments for yourself, run the following code in another file to generate the conformers after you have built the database. This process takes around 18 hours on a Macbook Pro M4 Pro Chip.


```python
from utils import Utils
import sqlite3

conn = sqlite3.connect("CCSMLDatabase.db")
cur = conn.cursor()

smiles = cur.execute("SELECT DISTINCT smi FROM master_clean").fetchall()

conformers_calculated = cur.execute("SELECT smi FROM conformers").fetchall()
conformer_smiles = set([smi[0] for smi in conformers_calculated])

smiles_reshaped = []
for smi in smiles:
    if smi not in conformer_smiles:
        smiles_reshaped.append(smi)

conformers = Utils().calculate_conformers("CCSMLDatabase.db", smiles_reshaped)
```

## Training Model

To train the model, make sure you have ``CCSMLDatabase.db`` in the same directory as ``train.py``. This process takes 10 minutes to train fulldataset on a M4 Pro Macbook Pro.

Run the following code in another file. To train the model without using METLIN data set ``use_metlin=False`` and to set a threshold for how many datapoints you want in a subclass in order to include it in the training and test sets,
set ``subclass_frequency_threshold=N`` where N is a number > 0.

```python
from train import CCSBase2

ccs_model = CCSBase2("CCSMLDatabase.db", 
                       "train_data.csv", 
                       "test_data.csv",
                       seed=26,
                       use_metlin=True, 
                       subclass_frequency_threshold=40)
ccs_model.fit()
ccs_model.predict()
```


## Inference

To perform inference, make sure the model ``ccsbase2.joblib`` and a inference csv file ``inference_input.csv`` is in your working directory with the exact column names "smi,ionmass,z,instrument,adduct"

Then run the file below.

```bash
python inference.py
```