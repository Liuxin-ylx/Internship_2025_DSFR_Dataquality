# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

from google.cloud import bigquery
from config.configuration import DatasetConfig
from config.obtainInfo import (
    obtain_table_name,
    load_check_rules
)
import pandas as pd 
import importlib
from typing import Tuple

def generate_clean_clause(schema, prefix: str = None) -> str:
    """Generate a cleaning clause to normalize all STRING columns.
        1. Multiple space --> single space
        2. Upper case
        3. Delete spaces before and after
        4. Remplace diacritics (accents)
        5. Handle NULL values

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
                f"r'\\s+', ' ')))"
            )
            full_expr = (
                f"CASE WHEN UPPER(TRIM({field_ref})) IN ('', 'NULL', 'N/A', 'NAN', 'NA', 'N.A!')"
                f"THEN NULL ELSE {clean_expr} END AS {field.name}"
            )
            clean_clause.append(full_expr)
        else:
            clean_clause.append(f"{field_ref} AS {field.name}")
    return ",\n    ".join(clean_clause)

def generate_clean_query(cfg:DatasetConfig, client:bigquery.Client, tableType:str) -> str:
    """
    Apply the cleaning clause to the table of the specified type (tableType).

    Args:
        tableType: Type of table to clean (raw, clean, excluded)
    """
    table = obtain_table_name(cfg, tableType)
    data = client.get_table(f"{cfg.project}.{cfg.dataset}.{table}")
    clean_clause_str = generate_clean_clause(data.schema, None)

    clean_query = f"""
    SELECT {clean_clause_str} 
    FROM `{cfg.project}.{cfg.dataset}.{table}`
    """
    return clean_query

def duplicate_rows_query(cfg:DatasetConfig, from_table, all_columns_clause:str) -> str:
    """
    Generate a query to find duplicate rows in the clean table.
    """
    # 1. Duplicate rows
    rule_row = f"""
    SELECT 'all_line_duplicate' AS reason, *
    FROM `{from_table}`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY {all_columns_clause}) > 1
    """
    
    return rule_row

def duplicate_keys_query(cfg:DatasetConfig, from_table:str) -> str:
    # 2. Duplicate keys
    rule_key = f"""
    SELECT 'primary_key_duplicate' AS reason, *
    FROM `{from_table}`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY {cfg.key_cle}) > 1
    """

    return rule_key

def barcode_length_query(cfg:DatasetConfig, from_table:str) -> str:
    # 3. Invalid barcode length
    rule_barcode = f"""
    SELECT 'wrong_barcode_length' AS reason, *
    FROM `{from_table}`
    WHERE LENGTH({cfg.main_barcode}) NOT IN (7, 8, 10, 13)
    """

    return rule_barcode

def date_format_query(cfg:DatasetConfig,from_table:str, schema:list) -> str:

    for field in schema:
        if field.field_type == "DATE":
            date_ref = f"{field.name}" 
            break
    else:
        raise ValueError("No DATE field found in the schema.")

        # 4. Invalid date format
    rule_date = f"""
    SELECT 'wrong_date' AS reason, *
    FROM `{from_table}`
    WHERE EXTRACT(YEAR FROM {date_ref}) < 1900 OR
          EXTRACT(YEAR FROM {date_ref}) > 2100 OR
          {date_ref} IS NULL
    """

    return rule_date


def generate_check_exclude_query(cfg:DatasetConfig, client:bigquery.Client) -> Tuple[str,str]:
    """
    Query1 : Check duplicates across the entire line
    Query2 : Check duplicates based on primary key
    Query3 : Check barcode length
    Query4 : Check date format
    This function generates a query to check for data quality issues in the clean table.
    """
    schema = client.get_table(f"{cfg.project}.{cfg.dataset}.{cfg.raw_table}").schema
    columns = [field.name for field in schema]
    all_columns_clause = ", ".join(columns)   

    clean_table = f"{cfg.project}.{cfg.dataset}.{cfg.clean_table}"
    excluded_table = f"{cfg.project}.{cfg.dataset}.{cfg.excluded_table}"
    
    # Combine all rules into a single query
    rule_queries = []
    active_rules = load_check_rules(cfg.dataset_type)
    if "duplicate_row" in active_rules:
        rule_queries.append(duplicate_rows_query(cfg, clean_table, all_columns_clause))
    if "duplicate_key" in active_rules:
        rule_queries.append(duplicate_keys_query(cfg, clean_table))
    if "barcode_length" in active_rules:
        rule_queries.append(barcode_length_query(cfg, clean_table))
    if "date_format" in active_rules:
        date_q = date_format_query(cfg, clean_table, schema)
        if date_q:
            rule_queries.append(date_q)

    union_query = " UNION ALL ".join(rule_queries)
    exclude_query = f"""
        WITH check_errors AS (
            {union_query}
        ),
        grouped AS (
            SELECT
                ARRAY_AGG(DISTINCT reason) AS reasons,
                {all_columns_clause}
            FROM check_errors
            GROUP BY {all_columns_clause}
        )
        SELECT * FROM grouped
    """

    filter_query = f"""
    SELECT * FROM `{clean_table}` 
    EXCEPT DISTINCT 
    SELECT {all_columns_clause} FROM `{excluded_table}`
    """

    return exclude_query,filter_query


def do_query_job(cfg:DatasetConfig,client:bigquery.Client, tableType:str, query:str) -> None:
    """
    Execute the query and save the result in the specified table
    """
    table = obtain_table_name(cfg, tableType)
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            destination = f"{cfg.project}.{cfg.dataset}.{table}",
            write_disposition = "WRITE_TRUNCATE",
        )
    )
    job.result()

def do_data2table_job(cfg: DatasetConfig, client:bigquery.Client, tableType:str, df:pd.DataFrame, schema:list) -> None:
    """
    Convert a dataframe to a table
    """
    table = obtain_table_name(cfg, tableType)
    job = client.load_table_from_dataframe(
        df,
        f"{cfg.project}.{cfg.dataset}.{table}",
        job_config=bigquery.LoadJobConfig(
            write_disposition = "WRITE_TRUNCATE",
            schema = schema
        )
    )
    job.result()