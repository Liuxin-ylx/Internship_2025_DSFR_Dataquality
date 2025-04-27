# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-04-25

import re
import pandas as pd
from google.cloud import bigquery

def standarize_by_frequence(df:pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    This function standardizes the brand names based on their frequency in the dataset.
    It uses a dictionary to map common brand names to their standardized form.
    """
    def normalize(brand):
        """
        Normalize the brand name by removing spaces and converting to uppercase.
        """
        if pd.isnull(brand):
            return None
        return re.sub(r'[A-Z0-9]', '',brand)
    
    def mapping(col:str):
        df["__normalized"] = df[col].apply(normalize)
        most_frequent = (
            df.groupby(["__normalized",col]).size()
            .reset_index(name="count")
            .sort_values(["__normalized", "count"], ascending=[True, False])
            .drop_duplicates(subset=["__normalized"])
            .set_index("__normalized")[col]
            .to_dict()
        )

        df[col] = df["__normalized"].map(most_frequent)
        print(pd.Series(most_frequent).value_counts())
        # pd.DataFrame.from_dict(most_frequent, orient="index", columns=['standardized'])


    for col in cols:
        mapping(col)
    df = df.drop(columns=["__normalized"])

    return df

