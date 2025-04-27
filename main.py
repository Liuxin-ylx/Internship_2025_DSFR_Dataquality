# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-04-24

from google.cloud import bigquery
from config.configuration import DatasetConfig
from modules.generateQuery import (
    generate_clean_query,
    generate_check_query,
    generate_exclude_query,
    do_query_job,
    do_data2table_job
)
from modules.standarize import standarize_by_frequence

class DataCleaningPipeline:
    def __init__(self, config: DatasetConfig):
        self.cfg = config
        self.client = bigquery.Client()
        self.schema = self.client.get_table(f"{self.cfg.project}.{self.cfg.dataset}.{self.cfg.raw_table}").schema

    def run(self):
        print("Step 1: Running cleaning query...")
        clean_query = generate_clean_query(self.cfg, self.client)
        do_query_job(self.cfg, self.client, self.cfg.clean_table, clean_query)

        print("Step 1.1: Standarize column values...")
        clean_data = self.client.query(f"""SELECT * FROM `{self.cfg.project}.{self.cfg.dataset}.{self.cfg.clean_table}`""").to_dataframe()
        target_cols = [field.name for field in self.schema if "brand" in field.name.lower() or "supplier" in field.name.lower()]
        clean_data = standarize_by_frequence(clean_data,target_cols)
        do_data2table_job(self.cfg, self.client, self.cfg.clean_table, clean_data, self.schema)

        print("Step 2: Running check query...")
        check_query = generate_check_query(self.cfg, self.client)
        do_query_job(self.cfg, self.client, self.cfg.clean_table, check_query)

        print("Step 3: Generating exclude query to identify removed rows...")
        exclude_query = generate_exclude_query(self.cfg, self.client)
        do_query_job(self.cfg, self.client, self.cfg.excluded_table, exclude_query)

        print("Pipeline finished.")


if __name__ == "__main__":    
    datacfg = DatasetConfig()
    pipeline = DataCleaningPipeline(datacfg)
    pipeline.run()
