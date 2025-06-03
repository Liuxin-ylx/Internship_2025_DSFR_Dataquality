# Architecture of the projects

```
.
|--- correction
|---  |   correction_dict.py
|---  |   __init__.py
|--- config
|---  |   configuration.py
|---  |   check_rules.yaml
|---  |   obtainInfo.py
|---  |   __init__.py
|---  |   __pycache__
|---  |   |   configuration.cpython-312.pyc
|---  |   |   __init__.cpython-312.pyc
|--- README.md
|--- main.py
|--- modules
|---  |   standarize.py
|---  |   generateEmbedding.py
|---  |   generateDictionary.py
|---  |   __init__.py
|---  |   __pycache__
|---  |   |   generateEmbedding.cpython-310.pyc
|---  |   |   nlp.cpython-310.pyc
|---  |   |   __init__.cpython-310.pyc
|---  |   nlp.py
|---  |   generateQuery.py
|--- data
|---  |   french_dictionary.txt
|---  |   dataset.csv               
```

# Explanation of pipeline
## Step1: Data cleaning
### 1.1 Format-level Cleaning
1. Convert multiple space to single space ```REGEXP_REPLACE(_, r'\\s+', ' ')```
2. Convert to upper case ```UPPER()```
3. Delete spaces before and after ```TRIM()```
4. Replace diacritics (accents) ```REGEXP_REPLACE(NORMALIZE(_, NFD), r'\pM', '')```
### 1.2 Semantic-level Cleaning
5. Handle missing values (convert variants like "N/A", "null", "" to NULL)
6. Unify the brands' names:

> To begin with, we remove all non-alphanumeric characters from brand names to normalize similar variants (e.g., "L'OREAL", "L OREAL" → "LOREAL").  
>
> Next, we group the original brand names based on their normalized form and count the frequency of each original name within each group.  
>
> We then select the most frequent original name in each group as the standardized brand name and store this mapping in a dictionary.
> 
> As a result, all variants like "L'OREAL" and "L OREAL" will be unified under the most common form depending on the frequency.  



## Step2: Data validaton
1. Check duplicates across the entire line
2. Check duplicates based on primary key
3. Check barcode length
4. [Option] Check date
