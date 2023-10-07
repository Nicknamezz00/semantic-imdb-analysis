### 专业实训作业

Basic idea: word2vec embedding + CNN, [here's the paper](./1408.5882.pdf)

This model (trained for 3 epochs) yields AUC = 0.96823 (on test data).

Ensemble of three convolutional networks (having different number of convolutional layers and feature maps) gives AUC = 0.97310.

#### Pre-trained model
* [3million * 300-dimension word2vec](https://code.google.com/archive/p/word2vec/)

#### Dataset
* [labeledTrainData](https://storage.googleapis.com/kagglesdsdata/competitions/3971/32703/labeledTrainData.tsv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1696907834&Signature=dM1XMkEr0ycu2CcZoWpQaF4T8mG1JLErm6p%2Bv%2B%2B0p5h0G50lAf7W8q22osyoz369zPqBNnUD1DFPH4jTNQFyHvm%2B2yFaCDj58fhvTM7ayv095jqDC3ju8uUTcctuRCU0xAwI2QSPqN6qAWuQrU4oTGiW739I%2BWhusDQrZZmT10%2BVoUSNS%2BSoT7WYy2Uj3BaLBPOVqER1PID3lZGZQcstEdbJkVP9KK6yJQ6B8b%2BwmnPu%2Feg3yTdyJk9qwAvkPPotaapn4kFv%2FTmAVP1tZSa5YkV4zflgriLTS2MYseZlTrbYTJ8qgoAiu4JQ0r5iRex7h065pb4UmGd36fLqcIA%2BwQ%3D%3D&response-content-disposition=attachment%3B+filename%3DlabeledTrainData.tsv.zip)
* [testData](https://storage.googleapis.com/kagglesdsdata/competitions/3971/32703/testData.tsv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1696907848&Signature=aNrSW%2FDrkAzO0rt8WtvKyQqMGkW%2B%2FgYP9ulI%2BJT0UAkyjCJhei9s7ZbAKDGk2vTSzOsLzWtq31YrlNW8GdXK7PtP65pavuY6v7zWmat6snWxnNxWn8%2F9yRqAhfapH8r%2F%2B%2Bq4KZMi0Ywj4R7wNuuIXZN4cpTx2F85jKr0u2E2KC4LFdbiYlDfqnN7dDPT3uDjkZygjKMvmwtLec0ro1es8rKelW5ot0vV4W5enZQ1rthgVwM3smrnBZBRWCopeZRKnlKXNX3bKVXg0aGu3kWRPODmRG278CBGhWCiXtNgOTXg6qs%2BEyxrZBjxE0kRYT9YdESJ7ZTpcYawtEY6e1meaw%3D%3D&response-content-disposition=attachment%3B+filename%3DtestData.tsv.zip)
* [unlabeledTrainData](https://storage.googleapis.com/kagglesdsdata/competitions/3971/32703/unlabeledTrainData.tsv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1696907861&Signature=EojgGewv6o6t%2FnLyT0cj4wNTqX0D367sEt1BkixZh3pVvWAx3X9D9jebssIZL6X1eYsY7crGlGJCRJdDUe%2B5xbPAh5akYXtd53tICsJRrnTycKrF58JMgeCN%2B5GSOtfWhK63c1JYZhLPKcl5LNnaWbfBeOFdXVRP7Oi3IJxy7Uw1qWNvCc%2F3ciaqO6DBDYoAstElo1RtNCdUr6%2BmUN6LHUP2zRqiysLy8CjzA%2BQ%2FmACg2VX0XtY%2B5oOR%2B1y%2F%2FwWTgUiOvpleK84jOioEwAqCJcz%2F3t%2BbwGdWmfsY0dlh7Zirl3PZugjSrdJcNgdgVeZuqwZCwxy7VJDosEv8%2FZBuLg%3D%3D&response-content-disposition=attachment%3B+filename%3DunlabeledTrainData.tsv.zip)

#### Quick start
* Create a clean environment
```shell
conda create --name env python=3.9
conda activate env
```

* Install dependencies 
```shell
pip3 install -r requirements.txt
```

* Download data 
* Prepare data
```shell
python3 ./code/process.py
```
* Train
```shell
python3 ./code/train.py
```
