# %%
%pip install polars
# %%
import os
import polars as pl
from datetime import datetime
# %%
SILVER_FILE = "nobel_prizes_silver.parquet"
GOLD_CATEGORY_FILE = "nobel_gold_category_summary.parquet"
GOLD_DECADE_FILE = "nobel_gold_decade_trends.parquet"
GOLD_MULTIPLE_WINNERS_FILE = "nobel_gold_multiple_winners.parquet"
# %% Funksjon for innlesing av aktive rader
def load_active_silver_data(silver_filepath: str) -> pl.DataFrame:
    if not os.path.exists(silver_filepath):
        raise FileNotFoundError(f"Finner ikke stien {silver_filepath}")

    df_silver = pl.read_parquet(silver_filepath)

    df_silver_active = df_silver.filter(pl.col("Silver_Active_Row") == 1)

    return df_silver_active

#%% Funksjon for beregning av prisbeløp per vinner
def enrich_with_calculated_amounts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Prize_Amount") * pl.col("Prize_Portion")).round(2).alias("Laureat_Prize_Amount"),
        (pl.col("Prize_Amount_Adjusted") * pl.col("Prize_Portion")).round(2).alias("Laureat_Prize_Amount_Adjusted")
    )

#%% Funksjon for kategori-oppsummering
def build_category_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("Category_Name")
        .agg(
            [
                pl.count("Laureat_Id").alias("Total_Laureates"),
                pl.col("Award_Year").n_unique().alias("Total_Award_Years"),
                pl.col("Award_Year").min().alias("First_Award_Year"),
                pl.col("Award_Year").max().alias("Latest_Award_Year"),
                pl.col("Laureat_Price_Amount")
                    .sum()
                    .round(0)
                    .alias("Total_Prize_Distributed_SEK"),
                pl.col("Laureat_Prize_Amount_Adjusted")
                    .sum()
                    .round(0)
                    .alias("Total_Adjusted_Prize_Distributed_SEK"),
                pl.col("Laureat_Prize_Amount_Adjusted")
                    .mean()
                    .round(0)
                    .alias("Avg_Adjusted_Prize_Per_Winner_SEK"),
            ]
        )
        .sort("Total_Laureates", descending=True)
        .with_columns(pl.lit(datetime.now())).alias("Gold_Loaded_Time")
    )

# %% Funksjon for tiårstrender
def build_decade_trends(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            ((pl.col("Award_Year") // 10) * 10).cast(pl.Int64).alias("Decade")
        )
        .group_by(["Decade", "Category_Name"])
        .agg(
            [
                pl.count("Laureat_Id").alias("Laureates_Count"),
                pl.col("Laureat_Prize_Amount_Adjusted")
                    .sum()
                    .round(0)
                    .alias("Decade_Total_Adjusted_Prize_SEK")
            ]
        )
        .sort(["Decade", "Category_Name"], descending=[True, False])
        .with_columns(pl.lit(datetime.now()).alias("Gold_Loaded_Time"))
    )

# %% Funksjon for flerdoble vinnere
def build_multiple_winners(df: pl.DataFrame) -> pl.DataFrame:
    multiple_ids = (
        df.group_by(["Laureat_Id", "Laureat_Name"])
        .agg(pl.count("Award_Year").alias("Prizes_Count"))
        .filter(pl.col("Prizes_Count") > 1)
        .select("Laureat_Id")
    )

    return (
        df.join(multiple_ids, on="Laureat_Id", how="inner")
        .select(
            [
                "Laureat_Id",
                "Laureat_Name",
                "Award_Year",
                "Category_Name",
                "Motivation"
            ]
        )
        .sort(["Laureat_Id", "Award_Year"])
        .with_columns(pl.lit(datetime.now().alias("Gold_Loaded_Time")))
    )

#%% Steg 1: Les inn aktive silver data
df_silver_active = load_active_silver_data(SILVER_FILE)

#%% Steg 2: Berik data med numerisk prisandel og beløp
df_enriched = enrich_with_calculated_amounts(df_silver_active)
df_enriched.select(
    [
        "Laureat_Name",
        "Prize_Portion",
        "Laureat_Prize_Amount"
    ]
).head(10)

#%%
df_test = df_enriched.group_by("Category_Name").agg([
    pl.count("Laureat_Id").alias("Total_Laureates"),
    pl.col("Award_Year").n_unique().alias("Total_Award_Years"),
    pl.col("Award_Year").min().alias("First_Award_Year"),
    pl.col("Award_Year").max().alias("Latest_Award_Year")
])
print(df_test)

#%%
df_test2 = df_enriched.filter(pl.col("Award_Year") >= 2000).group_by(["Category_Name", "Award_Year"]).agg(
    [
        pl.count("Laureat_Id").alias("Laureat_Count")
    ]
).sort(["Category_Name", "Award_Year"])
print(df_test2)

#%% Steg 3: Bygg og inspiser Gold Category Summary
df_gold_category = build_category_summary(df_enriched)
print(df_gold_category)

#%% Steg 4: Bygg og inspiser Gold Decade Trends
df_gold_decade = build_decade_trends(df_enriched)
print(df_gold_decade.head(15))

#%% Steg 5: Bygg og inspiser Gold Multiple Winners
df_gold_multiple = build_multiple_winners(df_enriched)
print(df_gold_multiple)

#%% Steg 6: Skriv alle Gold-tabeller til Parquet
df_gold_category.write_parquet(GOLD_CATEGORY_FILE)
df_gold_decade.write_parquet(GOLD_DECADE_FILE)
df_gold_multiple.write_parquet(GOLD_MULTIPLE_WINNERS_FILE)