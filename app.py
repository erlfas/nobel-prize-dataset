import os
import streamlit as st
import polars as pl
import plotly.express as px

st.set_page_config(page_title="Nobel Prize Explorer", page_icon="🏆", layout="wide")

SILVER_FILE = "nobel_prizes_silver.parquet"
GOLD_DECADE_FILE = "nobel_gold_decade_trends.parquet"
GOLD_CATEGORY_FILE = "nobel_gold_category_summary.parquet"

@st.cache_data
def load_parquet(filepath: str) -> pl.DataFrame:
    return pl.read_parquet(filepath)

st.title("🏆 Nobel Prize Data Explorer")
st.caption("Utforsk sølv- og gulllaget med interaktive filtre og Plotly-visualiseringer")

# --- Load Silver Data ---
try:
    df_silver = load_parquet(SILVER_FILE)
except Exception as e:
    st.error(f"Kunne ikke laste {SILVER_FILE}: {e}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filtre")

# Active rows filter
active_only = st.sidebar.checkbox("Kun aktive rader (Silver_Active_Row == 1)", value=True)
if active_only and "Silver_Active_Row" in df_silver.columns:
    df_filtered = df_silver.filter(pl.col("Silver_Active_Row") == 1)
else:
    df_filtered = df_silver

# Exclude unawarded / empty years if desired
skip_empty = st.sidebar.checkbox("Ekskluder uutdelte år (Laureat_Id er ikke null)", value=True)
if skip_empty and "Laureat_Id" in df_filtered.columns:
    df_filtered = df_filtered.filter(pl.col("Laureat_Id").is_not_null())

# Category filter
all_categories = sorted(df_silver["Category_Name"].drop_nulls().unique().to_list())
selected_categories = st.sidebar.multiselect("Velg kategorier", options=all_categories, default=all_categories)
if selected_categories:
    df_filtered = df_filtered.filter(pl.col("Category_Name").is_in(selected_categories))

# Year range slider
min_year = int(df_silver["Award_Year"].min())
max_year = int(df_silver["Award_Year"].max())
selected_years = st.sidebar.slider("Tidsperiode (År)", min_value=min_year, max_value=max_year, value=(min_year, max_year))
df_filtered = df_filtered.filter(
    (pl.col("Award_Year") >= selected_years[0]) & (pl.col("Award_Year") <= selected_years[1])
)

# Free text search
search_query = st.sidebar.text_input("Søk i vinnernavn eller motivasjon", "").strip()
if search_query:
    df_filtered = df_filtered.filter(
        pl.col("Laureat_Name").str.contains(f"(?i){search_query}")
        | pl.col("Motivation").str.contains(f"(?i){search_query}")
    )

# --- Top KPI Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rader vist", f"{len(df_filtered):,}")
col2.metric("Totalt i sølvfilen", f"{len(df_silver):,}")
unique_laureates = df_filtered["Laureat_Id"].n_unique() if "Laureat_Id" in df_filtered.columns else 0
col3.metric("Unike vinnere", f"{unique_laureates:,}")
col4.metric("Tidsspenn", f"{selected_years[0]} – {selected_years[1]}")

st.markdown("---")

# --- Tabs for Views ---
tab_charts, tab_table, tab_schema = st.tabs(["📊 Visualiseringer", "📋 Tabellvisning", "ℹ️ Skjema & Kolonner"])

with tab_charts:
    # -------------------------------------------------------------
    # 1. LINJEDIAGRAM: Utvikling i inflasjonsjustert premiebeløp over tiår
    # -------------------------------------------------------------
    st.subheader("📈 Utvikling i inflasjonsjustert premiebeløp over tiår")
    
    # Beregn tiårstrender dynamisk basert på filtre, eller bruk gull-filen
    use_gold_decade = os.path.exists(GOLD_DECADE_FILE) and len(selected_categories) == len(all_categories) and selected_years == (min_year, max_year)
    
    if os.path.exists(GOLD_DECADE_FILE):
        df_dec = load_parquet(GOLD_DECADE_FILE)
        if selected_categories:
            df_dec = df_dec.filter(pl.col("Category_Name").is_in(selected_categories))
        df_dec = df_dec.filter((pl.col("Decade") >= (selected_years[0] // 10) * 10) & (pl.col("Decade") <= (selected_years[1] // 10) * 10))
    else:
        # Fallback: Beregn direkte fra filtrert data
        df_dec = (
            df_filtered
            .with_columns(((pl.col("Award_Year") // 10) * 10).cast(pl.Int64).alias("Decade"))
            .group_by(["Decade", "Category_Name"])
            .agg([
                pl.count("Laureat_Id").alias("Laureates_Count"),
                (pl.col("Prize_Amount_Adjusted") * pl.col("Prize_Portion")).sum().round(0).alias("Decade_Total_Adjusted_Prize_SEK")
            ])
            .sort(["Decade", "Category_Name"])
        )

    if len(df_dec) > 0:
        fig_line = px.line(
            df_dec.to_pandas(),
            x="Decade",
            y="Decade_Total_Adjusted_Prize_SEK",
            color="Category_Name",
            markers=True,
            title="Inflasjonsjustert premiebeløp per tiår fordelt på kategori (SEK)",
            labels={
                "Decade": "Tiår",
                "Decade_Total_Adjusted_Prize_SEK": "Justert premiebeløp (SEK)",
                "Category_Name": "Kategori",
                "Laureates_Count": "Antall vinnere"
            },
            hover_data=["Laureates_Count"]
        )
        fig_line.update_layout(
            hovermode="x unified",
            xaxis=dict(tickmode="linear", dtick=10),
            yaxis=dict(title="Totalt justert premiebeløp (SEK)"),
            legend_title_text="Kategori"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Ingen tiårsdata tilgjengelig for valgte filtre.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. TREEMAP / SUNBURST: Fordeling av prispenger og vinnere
    # -------------------------------------------------------------
    st.subheader("🌳 Fordeling av prispenger og vinnere på tvers av kategorier")
    
    chart_col1, chart_col2 = st.columns([3, 1])
    with chart_col2:
        chart_type = st.radio("Velg diagramtype:", ["Treemap", "Sunburst"], horizontal=True)
        color_metric = st.selectbox(
            "Fargelegg etter:", 
            ["Total_Laureates", "Avg_Adjusted_Prize_Per_Winner_SEK"],
            format_func=lambda x: "Antall vinnere" if x == "Total_Laureates" else "Gj.snittlig premie per vinner (SEK)"
        )

    # Beregn eller hent kategorioppsummering
    if os.path.exists(GOLD_CATEGORY_FILE):
        df_cat = load_parquet(GOLD_CATEGORY_FILE)
        if selected_categories:
            df_cat = df_cat.filter(pl.col("Category_Name").is_in(selected_categories))
    else:
        df_cat = (
            df_filtered.group_by("Category_Name")
            .agg([
                pl.count("Laureat_Id").alias("Total_Laureates"),
                (pl.col("Prize_Amount_Adjusted") * pl.col("Prize_Portion")).sum().round(0).alias("Total_Adjusted_Prize_Distributed_SEK"),
                (pl.col("Prize_Amount_Adjusted") * pl.col("Prize_Portion")).mean().round(0).alias("Avg_Adjusted_Prize_Per_Winner_SEK")
            ])
        )

    if len(df_cat) > 0:
        cat_pandas = df_cat.to_pandas()
        
        with chart_col1:
            if chart_type == "Treemap":
                fig_cat = px.treemap(
                    cat_pandas,
                    path=["Category_Name"],
                    values="Total_Adjusted_Prize_Distributed_SEK",
                    color=color_metric,
                    color_continuous_scale="Blues",
                    title="Kategorifordeling (Størrelse = Totalt justert beløp i SEK)",
                    labels={
                        "Category_Name": "Kategori",
                        "Total_Adjusted_Prize_Distributed_SEK": "Totalt justert beløp (SEK)",
                        "Total_Laureates": "Antall vinnere",
                        "Avg_Adjusted_Prize_Per_Winner_SEK": "Gj.snitt beløp per vinner (SEK)"
                    },
                    hover_data={
                        "Total_Adjusted_Prize_Distributed_SEK": ":,.0f",
                        "Total_Laureates": True,
                        "Avg_Adjusted_Prize_Per_Winner_SEK": ":,.0f"
                    }
                )
            else:
                fig_cat = px.sunburst(
                    cat_pandas,
                    path=["Category_Name"],
                    values="Total_Adjusted_Prize_Distributed_SEK",
                    color=color_metric,
                    color_continuous_scale="Viridis",
                    title="Kategorifordeling (Sunburst: Størrelse = Totalt justert beløp)",
                    labels={
                        "Category_Name": "Kategori",
                        "Total_Adjusted_Prize_Distributed_SEK": "Totalt justert beløp (SEK)",
                        "Total_Laureates": "Antall vinnere",
                        "Avg_Adjusted_Prize_Per_Winner_SEK": "Gj.snitt beløp per vinner (SEK)"
                    },
                    hover_data={
                        "Total_Adjusted_Prize_Distributed_SEK": ":,.0f",
                        "Total_Laureates": True,
                        "Avg_Adjusted_Prize_Per_Winner_SEK": ":,.0f"
                    }
                )
            
            fig_cat.update_layout(margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Ingen kategoridata tilgjengelig for valgte filtre.")

with tab_table:
    st.subheader("Data View (Sølvtabell)")
    st.dataframe(df_filtered.to_pandas(), use_container_width=True)

with tab_schema:
    st.subheader("Skjema & Datatyper")
    schema_data = [{"Kolonne": col, "Datatype": str(dtype)} for col, dtype in df_silver.schema.items()]
    st.table(schema_data)
