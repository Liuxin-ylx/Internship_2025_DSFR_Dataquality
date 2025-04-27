from dataclasses import dataclass

@dataclass
class DatasetConfig:
    project: str = "lranalytics-eu-660531"
    dataset: str = "crf_liveramp_data_science_work"
    
    raw_table: str = "LIUXIN_crf_product_reference"
    clean_table: str = "LIUXIN_crf_product_reference_cleaned"
    excluded_table: str = "LIUXIN_crf_product_reference_excluded"

    key_cle: str = "country_id, barcode"
    main_barcode: str = "barcode"