# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-04-24

from google.cloud import bigquery
from config.configuration import DatasetConfig
import pandas as pd 

def generate_clean_clause(schema, prefix: str = None) -> str:
    """Generate a cleaning clause to normalize all STRING columns.
        1. Multiple space --> single space
        2. Upper case
        3. Delete spaces before and after
        4. Remplace diacritics (accents)

        Args:
        schema: BigQuery schema (list of SchemaField)
        prefix: Optional table alias prefix, e.g. "r"
    """
    clean_clause = []
    for field in schema:
        field_ref = f"{prefix}.{field.name}" if prefix else field.name

        if field.field_type == "STRING":
            clean_expr = (
                f"TRIM(UPPER(REGEXP_REPLACE("
                f"REGEXP_REPLACE(NORMALIZE({field_ref}, NFD), r'\pM', ''), "
                f"r'\\s+', ' '))) AS {field.name}"
            )
            clean_clause.append(clean_expr)
        else:
            clean_clause.append(field_ref + f" AS {field.name}")
    return ",\n    ".join(clean_clause)


def generate_clean_query(cfg:DatasetConfig, client:bigquery.Client) -> str:
    """
    Query1 : Simple cleaning
    """
    data = client.get_table(f"{cfg.project}.{cfg.dataset}.{cfg.raw_table}")
    clean_clause_str = generate_clean_clause(data.schema, None)

    clean_query = f"""
    SELECT {clean_clause_str} 
    FROM `{cfg.project}.{cfg.dataset}.{cfg.raw_table}`
    """
    return clean_query

def generate_check_query(cfg:DatasetConfig, client:bigquery.Client) -> str:
    """
    Query1 : Check duplicates across the entire line
    Query2 : Check duplicates based on primary key
    Query3 : Check barcode length
    Query4 : Check consistency of barcode with hierarchy
    """
    data = f"{cfg.project}.{cfg.dataset}.{cfg.raw_table}"
    schema = client.get_table(data).schema
    
    
    columns = [field.name for field in schema]
    all_columns_clause = ", ".join(columns)   
    
    hierarchy_cols = [field.name for field in schema if "hierarchy" in field.name.lower()]
    hierarchy_clause = " || '|' || ".join(hierarchy_cols)

    check_query = f"""
    WITH doublon AS (
        SELECT *, ROW_NUMBER() 
        OVER (PARTITION BY {all_columns_clause}) AS doublons 
        FROM `{cfg.project}.{cfg.dataset}.{cfg.clean_table}`
    ),
    query1 AS (
        SELECT * FROM doublon 
        WHERE doublons = 1
    ),
    unique_cle AS (
        SELECT *, ROW_NUMBER() 
        OVER (PARTITION BY {cfg.key_cle} ORDER BY {cfg.key_cle}) AS cles 
        FROM query1
    ),
    query2 AS (
        SELECT * FROM unique_cle 
        WHERE cles = 1
    ),
    query3 AS (
        SELECT * FROM query2 
        WHERE LENGTH({cfg.main_barcode}) IN (8,13,128)
    ),
    bad_barcodes AS (
        SELECT {cfg.main_barcode} FROM query3
        GROUP BY {cfg.main_barcode}
        HAVING COUNT(DISTINCT CONCAT({hierarchy_clause})) > 1
    ),
    query4 AS (
        SELECT * FROM query3
        WHERE {cfg.main_barcode} NOT IN (
            SELECT {cfg.main_barcode} FROM bad_barcodes
        )
    )
    SELECT {all_columns_clause} FROM query4
    """
    return check_query


def generate_exclude_query(cfg:DatasetConfig, client:bigquery.Client) -> str:
    """
    Query : Exclude the cleaned table from the original table
    """
    data = client.get_table(f"{cfg.project}.{cfg.dataset}.{cfg.raw_table}")
    clean_clause_str = generate_clean_clause(data.schema,"r")

    keys = cfg.key_cle.split(", ")
    join_condition = " AND ".join([f"r.{k} = c.{k}" for k in keys])
    null_condition = " AND ".join([f"c.{k} IS NULL" for k in keys])

    exclude_query = f"""
    SELECT {clean_clause_str} FROM `{cfg.project}.{cfg.dataset}.{cfg.raw_table}` r
    LEFT JOIN `{cfg.project}.{cfg.dataset}.{cfg.clean_table}` c
    ON {join_condition}
    WHERE {null_condition}
    """
    return exclude_query

def do_query_job(cfg:DatasetConfig,client:bigquery.Client, table:str, query:str) -> None:
    """
    Execute the query and save the result in the specified table
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            destination = f"{cfg.project}.{cfg.dataset}.{table}",
            write_disposition = "WRITE_TRUNCATE",
        )
    )
    job.result()

def do_data2table_job(cfg: DatasetConfig, client:bigquery.Client, table:str, df:pd.DataFrame, schema:list) -> None:
    """
    Convert a dataframe to a table
    """
    job = client.load_table_from_dataframe(
        df,
        f"{cfg.project}.{cfg.dataset}.{table}",
        job_config=bigquery.LoadJobConfig(
            write_disposition = "WRITE_TRUNCATE",
            schema = schema,
        )
    )
    job.result()