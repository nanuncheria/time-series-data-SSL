# Self-Supervised Learning of time series data

## T_CPC.py
model with temporal encodings only
- generally retains the model examined in the referenced paper

## TS_CPC.py
model with temporal spectral encodings; suggested new model for improvement
- incorporates the use of spectral encodings alongside temporal encodings

## S_CPC.py
model with spectral encodings only
- speculates the effectiveness of spectral encodings


## unsupervised.py
implements two different linear classification methods onto the unsupervised training results of the model
produces relevant metrics to evaluate the training quality

## supervised.py
implements likewise onto the supervised training results

## before training.ipynb
no additional training done to determine the effectiveness of initial training

## exercise provided.ipynb
exercise sheet provided by the group responsible for the codes used in the referenced paper

## un-sup comparison.ipynb
codes used to produce graph that compares the unsupervised and supervised model results




### UNSUPERVISED.csv - data from unsupervised.py
### UNSUPERVISED_LR.csv - only Logistic Regression results remaining, ClassifierNN results omitted
### SUPERVISED.csv - data from supervised.py
### withouttrain.csv - data from before training.ipynb
data is saved within these files to produce graphs that portray ROC-AUC results over epochs for diff conditions



