# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-13

from google.cloud import bigquery
from config.configuration import DatasetConfig
from config.table import obtain_dataframe
from modules.standarize import standarize_by_frequence
from modules.generateQuery import (
    generate_clean_query,
    generate_check_exclude_query,
    do_query_job,
    do_data2table_job
)



class DataCleaningPipeline:
    def __init__(self, config: DatasetConfig):
        self.cfg = config
        self.client = bigquery.Client()
        self.schema = self.client.get_table(f"{self.cfg.project}.{self.cfg.dataset}.{self.cfg.raw_table}").schema

    def run(self):
        print("Step 1: Data Cleaning...")
        print("Step 1.1: Running Format-level Cleaning...")
        clean_query = generate_clean_query(self.cfg, self.client, "raw")
        do_query_job(self.cfg, self.client, "clean", clean_query)

        print("Step 1.2: Running Semantic-level Cleaning...")
        print("--------> Handling missing values...")
        print("--------> Standarize column values...")
        clean_data = obtain_dataframe(self.cfg, self.client, "clean")
        target_cols = [field.name for field in self.schema if "brand" in field.name.lower() or "supplier" in field.name.lower()]
        clean_data = standarize_by_frequence(clean_data,target_cols)
        do_data2table_job(self.cfg, self.client, "clean", clean_data,self.schema)

        print("Step 2: Data validation...")
        exclude_query,final_query = generate_check_exclude_query(self.cfg, self.client)
        do_query_job(self.cfg, self.client, "excluded", exclude_query)
        do_query_job(self.cfg, self.client, "clean", final_query)

        print("Pipeline finished.")


if __name__ == "__main__":    
    datacfg = DatasetConfig()
    pipeline = DataCleaningPipeline(datacfg)
    pipeline.run()
