# anonymization
Implementation of best practice methods for data anonymization.

## Getting started

Set up Conda environment Python 3.11 (anonymization311)

```conda create -n anonymization311 python=3.11```

Look, where conda environment is stored

```conda info --envs```

Activate the environment in Terminal

```conda activate anonymization311```

Install requirements 

```pip3 install -r requirements.txt```

Download datasets

1. create folder datasets
2. download and extract https://archive.ics.uci.edu/static/public/2/adult.zip
3. add folders in the created folder datasets (e.g. datasets/adult)
4. in datasets/adult/adult.data add ''age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
'' in first line 
5. in datasets/adult/adult.test add ''age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
'' in second line 

## Reproducibility


## Documentation


## Acknowledgements

The research project EAsyAnon (“Verbundprojekt: Empfehlungs- und Auditsystem zur Anonymisierung”, funding indicator: 16KISA128K) is funded by the European Union under the umbrella of the funding guideline “Forschungsnetzwerk Anonymisierung für eine sichere Datennutzung” from the German Federal Ministry of Education and Research (BMBF).
