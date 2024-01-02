# anonymization
Implementation and overview of methods for data anonymization. The implementations restrict on tabular data
such as microdata. The repository is work in process in the project **[EAsyAnon](#acknowledgements)**.

## Getting started

Set up a Conda environment with Python 3.11 (anonymization311)

```bash
conda create -n anonymization311 python=3.11
```

Look, where this conda environment is stored

```bash
conda info --envs
```

Activate the environment in Terminal

```bash
conda activate anonymization311
```

Install requirements 

```bash
pip3 install -r requirements.txt
```

Download datasets

1. create folder datasets
2. download and extract https://archive.ics.uci.edu/static/public/2/adult.zip
3. add folders in the created folder datasets (e.g. datasets/adult)
4. in datasets/adult/adult.data add ''age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
'' in first line 
5. in datasets/adult/adult.test add ''age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
'' in second line

## Tests

```bash
python -m unittest
```

## Further Reading

If you find this repository useful please consider to cite us. The following work deals with the **technical and legal** aspects 
when anonymizing tabular data (https://www.mdpi.com/2078-2489/14/9/487).

```bibtex
@Article{info14090487,
    AUTHOR = {Aufschläger, Robert and Folz, Jakob and März, Elena and Guggumos, Johann and Heigl, Michael and Buchner, Benedikt and Schramm, Martin},
    TITLE = {Anonymization Procedures for Tabular Data: An Explanatory Technical and Legal Synthesis},
    JOURNAL = {Information},
    VOLUME = {14},
    YEAR = {2023},
    NUMBER = {9},
    ARTICLE-NUMBER = {487},
    URL = {https://www.mdpi.com/2078-2489/14/9/487},
    ISSN = {2078-2489},
    ABSTRACT = {In the European Union, Data Controllers and Data Processors, who work with personal data, have to comply with the General Data Protection Regulation and other applicable laws. This affects the storing and processing of personal data. But some data processing in data mining or statistical analyses does not require any personal reference to the data. Thus, personal context can be removed. For these use cases, to comply with applicable laws, any existing personal information has to be removed by applying the so-called anonymization. However, anonymization should maintain data utility. Therefore, the concept of anonymization is a double-edged sword with an intrinsic trade-off: privacy enforcement vs. utility preservation. The former might not be entirely guaranteed when anonymized data are published as Open Data. In theory and practice, there exist diverse approaches to conduct and score anonymization. This explanatory synthesis discusses the technical perspectives on the anonymization of tabular data with a special emphasis on the European Union&rsquo;s legal base. The studied methods for conducting anonymization, and scoring the anonymization procedure and the resulting anonymity are explained in unifying terminology. The examined methods and scores cover both categorical and numerical data. The examined scores involve data utility, information preservation, and privacy models. In practice-relevant examples, methods and scores are experimentally tested on records from the UCI Machine Learning Repository&rsquo;s &ldquo;Census Income (Adult)&rdquo; dataset.},
    DOI = {10.3390/info14090487}
}
```

## Acknowledgements

The research project **EAsyAnon** (“Verbundprojekt: Empfehlungs- und Auditsystem zur Anonymisierung”, funding indicator: 16KISA128K) is funded by the European Union under the umbrella of the funding guideline “Forschungsnetzwerk Anonymisierung für eine sichere Datennutzung” from the German Federal Ministry of Education and Research (BMBF).
