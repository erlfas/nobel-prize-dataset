# %%
%pip install polars requests
# %%
import polars as pl
import os
from datetime import datetime

#%%
BRONZE_FILE = "nobel_prizes_bronze.parquet"
SILVER_FILE = "nobel_prizes_silver.parquet"

# %%
def process_silver_scd2(df_bronze: pl.DataFrame, silver_filepath: str) -> pl.DataFrame:
    # 1. Klargjør datatyper
    reference_year = (
        df_bronze.filter(
            (pl.col("Prize_Amount") == pl.col("Prize_Amount_Adjusted")) & 
            (pl.col("Prize_Amount") > 0)
        )
        .select(pl.col("Award_Year").cast(pl.Int64).max())
        .item()
    )

    df_bronze_prepared = df_bronze.with_columns(
        pl.col("Award_Year").cast(pl.Int64),
        pl.col("Laureat_Id").cast(pl.Int64),
        pl.col("Date_Awarded").cast(pl.Date),
        pl.lit(reference_year).cast(pl.Int64).alias("Prize_Adjusted_Reference_Year"),
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

#%%
if __name__ == "__main__":
    df_bronze = pl.read_parquet(BRONZE_FILE)
    df_silver = process_silver_scd2(df_bronze, SILVER_FILE)
    df_silver.write_parquet(SILVER_FILE)
    df_silver_read = pl.read_parquet(SILVER_FILE)
    print(df_silver_read.sort("Silver_Updated_Time", descending=True).head(10))
