# %%
import time
import requests
import polars as pl
import hashlib
from datetime import datetime
BASE_URL = "https://api.nobelprize.org/2.1/nobelPrizes"
LIMIT = 100
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
BRONZE_FILE = "nobel_prizes_bronze.parquet"

# %% Ingestion: Hente rådata fra API
def fetch_all_nobel_prizes() -> list[dict]:
    """Henter alle nobelpriser paginert fra API-et."""
    offset = 0
    all_prizes = []
    total_count = None
    while True:
        url = f"{BASE_URL}?offset={offset}&limit={LIMIT}"
        print(f"Kaller url {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if total_count is None:
            total_count = data.get("meta", {}).get("count", 0)
            print(f"Totalt antall nobelpriser å hente: {total_count}")
        prizes = data.get("nobelPrizes", [])
        if prizes:
            all_prizes.extend(prizes)
        offset += LIMIT
        if offset >= total_count or not prizes:
            break
        time.sleep(0.1)
    return all_prizes

#%% Flate ut nøstet JSON
def flatten_nobel_prizes(raw_prizes: list[dict]) -> pl.DataFrame:
    df = pl.from_dicts(raw_prizes)
    return (
        df
        .with_columns(
            pl.col("category").struct.field("en").alias("category_name"),
            pl.col("categoryFullName").struct.field("en").alias("category_full_name")
        )
        .explode("laureates")
        .with_columns(
            pl.col("laureates").struct.field("id").alias("laureat_id"),
            pl.coalesce([
                pl.col("laureates").struct.field("knownName").struct.field("en"),
                pl.col("laureates").struct.field("orgName").struct.field("en")
            ]).alias("laureat_name"),
            pl.col("laureates").struct.field("portion").alias("prize_portion"),
            pl.col("laureates").struct.field("motivation").struct.field("en").alias("motivation")
        )
        .select([
            pl.col("awardYear").alias("Award_Year"),
            pl.col("dateAwarded").alias("Date_Awarded"),
            pl.col("prizeAmount").alias("Prize_Amount"),
            pl.col("prizeAmountAdjusted").alias("Prize_Amount_Adjusted"),
            pl.col("laureat_id").alias("Laureat_Id"),
            pl.col("laureat_name").alias("Laureat_Name"),
            pl.col("prize_portion").alias("Prize_Portion"),
            pl.col("motivation").alias("Motivation"),
            pl.col("category_name").alias("Category_Name"),
            pl.col("category_full_name").alias("Category_Full_Name")
        ])
        .filter(pl.col("Laureat_Id").is_not_null())
    )

#%% Legg til systemkolonner
def enrich_bronze_data(df_flat: pl.DataFrame) -> pl.DataFrame:
    key_cols = ["Laureat_Id", "Award_Year", "Category_Name"]
    data_cols = [
        "Laureat_Name", 
        "Date_Awarded", 
        "Prize_Amount", 
        "Prize_Amount_Adjusted", 
        "Prize_Portion", 
        "Motivation", 
        "Category_Name",
        "Category_Full_Name"
    ]

    return (
        df_flat
        .with_columns(
            pl.lit(datetime.now()).alias("Bronze_Loaded_Time"),

            pl.concat_str(
                [pl.col(c).cast(pl.Utf8).fill_null("_NULL_") for c in data_cols], separator="|"
            ).map_elements(
                lambda x: hashlib.md5(x.encode("utf-8")).hexdigest(),
                return_dtype=pl.Utf8
            ).alias("Data_Key"),
            pl.concat_str(
                [pl.col(c).cast(pl.Utf8) for c in key_cols],
                separator="|"
            ).alias("Identifier_Key")
        )
        .select([
            "Identifier_Key",
            "Data_Key",
            "Bronze_Loaded_Time",
            "Laureat_Id", 
            "Award_Year", 
            "Category_Name",
            "Category_Full_Name",
            "Laureat_Name", 
            "Date_Awarded", 
            "Prize_Amount", 
            "Prize_Amount_Adjusted", 
            "Prize_Portion", 
            "Motivation"
        ])
    )

#%% Kjør hele pipelinen
def run_pipeline(output_filepath: str = BRONZE_FILE) -> pl.DataFrame:
    raw_prizes = fetch_all_nobel_prizes()

    df_flat = flatten_nobel_prizes(raw_prizes)

    df_bronze = enrich_bronze_data(df_flat)

    df_bronze.write_parquet(output_filepath)

    return df_bronze

#%% Kjør
if __name__ == "__main__":
    df_bronze = run_pipeline()
    print(df_bronze.sort("Bronze_Loaded_Time", descending=True).head(10))