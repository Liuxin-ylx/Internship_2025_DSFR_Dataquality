# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-13

from config.configuration import DatasetConfig
from google.cloud import bigquery
import pandas as pd

def obtain_table_name(cfg:DatasetConfig, tableType: str) -> str:
    
    if tableType == "raw":
        table = cfg.raw_table
    elif tableType == "clean":
        table = cfg.clean_table
    elif tableType == "excluded":
        table = cfg.excluded_table
    else:
        raise ValueError("Invalid table type. Choose from 'raw', 'clean', or 'excluded'.")
    return table

def obtain_dataframe(cfg:DatasetConfig,client:bigquery.Client, tableType: str) -> pd.DataFrame:

    table = obtain_table_name(cfg, tableType)
    data = client.query(f"""
                        SELECT * FROM `{cfg.project}.{cfg.dataset}.{table}`
                        """).to_dataframe()
    
    return data