# %%
%pip install polars requests
# %%
import time
import requests
import polars as pl
import hashlib
import os
from datetime import datetime

BASE_URL = "https://api.nobelprize.org/2.1/nobelPrizes"
LIMIT = 100

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# %%
def fetch_all_nobel_prizes():
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

        if prizes and len(prizes) > 0:
            all_prizes.extend(prizes)

        offset += LIMIT

        if offset >= total_count or not prizes:
            print(f"Avbryter fordi offsett er {offset} og total_count er {total_count} og prizes har lengde {len(prizes)}")
            break

        time.sleep(0.1)

    return all_prizes

# %%
raw_prizes = fetch_all_nobel_prizes()
df = pl.from_dicts(raw_prizes)

print(f"Hentet totalt {len(df)} rader inn i Polars dataframe")
# %% ETL transformasjon
df_flat = (
    df
    .with_columns(
        pl.col("category").struct.field("en").alias("category_name"),
        pl.col("categoryFullName").struct.field("en").alias("category_full_name")
    )
    .explode("laureates")
    .with_columns(
        pl.col("laureates").struct.field("id").alias("laureat_id"),
        pl.col("laureates").struct.field("knownName").struct.field("en").alias("laureat_name"),
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
)
# %% Legger til systemkolonner
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

df_flat_enriched = (
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

# %%
print(df_flat_enriched.head(10))

# %%
BRONZE_FILE = "nobel_prizes_bronze.parquet"

# %%
df_flat_enriched.write_parquet(BRONZE_FILE)

# %%
df_bronze = pl.read_parquet(BRONZE_FILE)

print(f"Lest inn bronze: {df_bronze.height} og {df_bronze.width}")

# %%
def process_silver_scd2(df_bronze: pl.DataFrame, silver_filepath: str) -> pl.DataFrame:
    # 1. Klargjør datatyper
    df_bronze_prepared = df_bronze.with_columns(
        pl.col("Award_Year").cast(pl.Int64),
        pl.col("Laureat_Id").cast(pl.Int64),
        pl.col("Date_Awarded").cast(pl.Date),
        pl.when(pl.col("Prize_Portion").str.contains("/"))
            .then(
                pl.col("Prize_Portion").str.split("/").list.get(0).cast(pl.Float64)
                / pl.col("Prize_Portion").str.split("/").list.get(1).cast(pl.Float64)
            )
            .otherwise(
                pl.col("Prize_Portion").cast(pl.Float64)
            )
            .alias("Prize_Portion")
    )

    # 2. Ved første gangs opprettelse
    if not os.path.exists((silver_filepath)):
        print("Initialiserer ny Silver-tabell (Første SCD2-kjøring)...")
        return (
            df_bronze_prepared
            .with_columns([
                pl.col("Bronze_Loaded_Time").alias("Silver_From_Time"),
                pl.lit(None).cast(pl.Datetime).alias("Silver_To_Time"),
                pl.lit(1).cast(pl.Int8).alias("Silver_Active_Row"),
                pl.lit(datetime.now()).alias("Silver_Updated_Time")
            ])
            .sort(["Award_Year", "Category_Name"], descending=[True, False])
        )

    # 3. Ved etterfølgende kjøringer: Sammenlign eksisterende silver med innkommende data
    df_silver_existing = pl.read_parquet(silver_filepath)

    # Del 3a Deaktiverte historiske rader beholdes urørt
    df_silver_existing_inactive = df_silver_existing.filter(pl.col("Silver_Active_Row") == 0)

    # Del aktiver rader fra silver
    df_silver_existing_active = df_silver_existing.filter(pl.col("Silver_Active_Row") == 1)

    # Sammenstill aktive rader mot innkommende bronze basert på Identifier_Key
    joined = df_silver_existing_active.join(
        df_bronze_prepared.select([
            "Identifier_Key",
            "Data_Key",
            pl.col("Bronze_Loaded_Time").alias("Incoming_Bronze_Loaded_Time")
        ]),
        on="Identifier_Key",
        how="left",
        suffix="_incoming"
    )

    # Aktive & uendret (Data_Key er lik): Bevar som aktiv
    # 3b
    df_silver_existing_active_unchanged = joined.filter(
        pl.col("Data_Key_incoming").is_not_null() &
        (pl.col("Data_Key") == pl.col("Data_Key_incoming"))
    ).select(df_silver_existing_active.columns)

    # Aktive & endret (Data_Key er ulik): Lukk eksisterende rad
    # Silver_To_Time := Bronze_Loaded_Time og Silver_Active_Row = 0
    # 3c
    df_silver_existing_active_changed_closed = joined.filter(
        pl.col("Data_Key_incoming").is_not_null() &
        (pl.col("Data_Key") != pl.col("Data_Key_incoming"))
    ).with_columns([
        pl.col("Incoming_Bronze_Loaded_Time").alias("Silver_To_Time"),
        pl.lit(0).cast(pl.Int8).alias("Silver_Active_Row"),
        pl.lit(datetime.now()).alias("Silver_Updated_Time")
    ]).select(df_silver_existing_active.columns)

    # Nye eller oppdaterte rader fra innkommende bronze som skal
    # settes inn som nye aktive rader
    joined_bronze = df_bronze_prepared.join(
        df_silver_existing_active.select(["Identifier_Key", "Data_Key"]),
        on="Identifier_Key",
        how="left",
        suffix="_existing"
    )

    # 3d
    df_bronze_new_or_changed = joined_bronze.filter(
        pl.col("Data_Key_existing").is_null() | (pl.col("Data_Key") != pl.col("Data_Key_existing"))
    ).with_columns([
        pl.col("Bronze_Loaded_Time").alias("Silver_From_Time"),
        pl.lit(None).cast(pl.Datetime).alias("Silver_To_Time"),
        pl.lit(1).cast(pl.Int8).alias("Silver_Active_Row"),
        pl.lit(datetime.now()).alias("Silver_Updated_Time")
    ]).select(df_silver_existing_active.columns)

    return pl.concat([
        df_silver_existing_inactive,
        df_silver_existing_active_unchanged,
        df_silver_existing_active_changed_closed,
        df_bronze_new_or_changed
    ]).sort(["Award_Year", "Category_Name", "Silver_From_Time"], descending=[True, False, False])



# %% 
SILVER_FILE = "nobel_prizes_silver.parquet"
df_silver = process_silver_scd2(df_bronze, SILVER_FILE)

# %% Skriv til silver
df_silver.write_parquet(SILVER_FILE)

# %% Verifiser silver
df_silver_read = pl.read_parquet(SILVER_FILE)
print(f"Silver er oppdatert og har nå {df_silver_read.height} antall rader")
print(df_silver_read.select([
    "Identifier_Key",
    "Data_Key",
    "Silver_From_Time",
    "Silver_To_Time",
    "Silver_Active_Row"
]).head(10))
# %%
print(df_silver_read.filter(pl.col("Category_Name").is_null()).head(100))
