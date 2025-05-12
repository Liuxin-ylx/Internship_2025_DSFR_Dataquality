# Architecture of the projects

```
My_project/ <br/>
├── main.py <br/>                 
├── config/                  # Configuration, parameters
│   └── __init__.py       
│   └── configuration.py       
├── modules/                 # Functions
│   ├── __init__.py         
│   ├── generateQuery.py        
│   ├── ...                  
├── data/                    # Data（.csv/.json）
└── README.md                
```

# Explanation of pipeline
## Step1: Simple cleaning
### SQL
1. Multiple space --> single space
2. Convert to upper case
3. Delete spaces before and after
4. Replace diacritics (accents)
### Python
5. Unifier les noms des marques

## Step2: Simple check
### SQL
1. Check duplicates across the entire line
2. Check duplicates based on primary key
3. [Option] Check barcode length
4. [Option] Check consistency of barcode with hierarchy
