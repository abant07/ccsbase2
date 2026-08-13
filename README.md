# CCSBase2

A Collision Cross Section prediction model using a XGBoostRegressor model.

## Documentation

- **Fingerprint explorer** — hosted on Streamlit: [https://ccsbase2-fingerprint-explorer.streamlit.app/](https://ccsbase2-fingerprint-explorer.streamlit.app/)

## Dependencies

### 1. Create a Conda environment (Python 3.12)

```bash
conda create -n ccsbase2 python=3.12 -y
conda activate ccsbase2
```

### 2. Install
```bash
pip install numpy pandas scikit-learn rdkit xgboost joblib requests matplotlib streamlit shap
```

## Building Database

### Build Main Database
CCSbase2 aggregates across the 5 different datasets listed below. AllCCS was obtained by downloading chemicals manually using their online database.


#### 
- [CCSBase](https://ccsbase.net/)
- [PNNL](https://pnnl-comp-mass-spec.github.io/MetabolomicsCCS/)
- [ALLCCS](http://allccs.zhulab.cn/database/browser)
- [METLIN-CCS](https://www.dropbox.com/scl/fi/9xctm5ub834muw1qrvd5b/CCS-Publication-V3.zip?e=2&file_subpath=%2FCCS-Publication-V3&rlkey=zi9xaua4zzgpiiaznexabpg7i&dl=0)
- [Dataset found from ACS publication](https://pubs.acs.org/doi/10.1021/acs.jafc.2c00724)


#### [Download Model Weights](https://drive.google.com/file/d/17zLr5OTGReIVL19vkkq5W-RAJ4tJ6mu7/view?usp=sharing)


Please note that building the database from scratch takes a very long time as thousands of API calls need to be made. Instructions have been given below.

Run the code below in another file. We advise you to use Google Colab due to PubChem rate limiting IP addresses which results in a "ServerBusy" error. You may need to just call ``ccsml.find_smiles()`` multiple times while commenting out the rest of the method calls.

After determining as many SMILES, call ``ccsml.find_inchikey()`` at the end which calls PubChem API to retrieve InChiKey when given SMILES string. This is to obtain InChiKeys for CCSbase datapoints since CCSbase stores SMILES, but not InChiKeys. 

ClassyFire also does rate limiting, so call ``ccsml.find_inchikey()`` multiple times while commenting out all other method calls. NOTE: APIs will not find SMILES string and/or subclass for all chemicals.

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
ccsml.build_ood_dataset("ood_testset.csv")
```


### Classify Unknown Subclasses

ClassyFire will not identify subclasses for all chemicals, so a XGBClassifier was developed to classify the unknown subclasses. All subclasses that are known are grouped and groups less than less than 30 chemicals are not included during training and act as a proxy out of distribution (OOD) dataset.

```python
from classifier import SubclassClassifier

classifier = SubclassClassifier(
    "CCSMLDatabase.db", seed=26, fp_vocab_file="ccsbase2_fp_vocab.joblib",
    min_subclass_count=30, novelty_threshold=0.7,
    n_estimators=[5000, 5500, 6000],
    max_depth=[8, 9],
    learning_rate=[0.03, 0.05],
    subsample=[0.9],
    colsample_bytree=[0.9],
    reg_lambda=[30],
    gamma=[1],
    min_child_weight=[5],
)

classifier.fit()
classifier.eval() # evaluates on proxy OOD set
classifier.predict() # performes inference on compounds with unknown subclasses
```


## Training Model

Run the following code in another file. Set ``use_metlin=False`` to train without METLIN dataset.

```python
from train import CCSBase2

ccs_model = CCSBase2("CCSMLDatabase.db",
                    n_estimators=[6000, 7000, 8000, 10000, 12000, 14000, 20000],
                    max_depth=[8, 10, 12, 13, 15],
                    learning_rate=[0.01, 0.02, 0.03],
                    subsample=[0.9],
                    colsample_bytree=[0.5, 0.9],
                    reg_lambda=[30],
                    min_child_weight=[1],
                    gamma=[1],
                    seed=26,
                    use_metlin=True,
                    fp_vocab_file="ccsbase2_fp_vocab.joblib"
                )
ccs_model.fit()
ccs_model.eval() # evaluates on test_data.csv
```


## Inference

To perform inference, create a `.csv` file with column names ``smi,adduct`` and use the `CCSBase2` `predict()` method to pass in your .csv file. Optionally, if you're looking to evaluate the model on you're own dataset, pass the column name in your `.csv` that holds the ground truth CCS as a second parameter to `predict()`.

```bash
ccs_model = CCSBase2("CCSMLDatabase.db",
                    n_estimators=[6000, 7000, 8000, 10000, 12000, 14000, 20000],
                    max_depth=[8, 10, 12, 13, 15],
                    learning_rate=[0.01, 0.02, 0.03],
                    subsample=[0.9],
                    colsample_bytree=[0.5, 0.9],
                    reg_lambda=[30],
                    min_child_weight=[1],
                    gamma=[1],
                    seed=26,
                    use_metlin=True,
                    fp_vocab_file="ccsbase2_fp_vocab.joblib"
                )

ccs_model.predict("./datasets/ood_testset.csv", "ccs")
```