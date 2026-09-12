import plotly.express as px
import polars as pl
import streamlit as st

st.title("Totalt justert premiebeløp")

CATEGORY_MAPPING = {
    "Chemistry": "Kjemi",
    "Literature": "Litteratur",
    "Peace": "Fred",
    "Physics": "Fysikk",
    "Physiology or Medicine": "Fysiologi eller medisin",
    "Economic Sciences": "Økonomi",
}

@st.cache_data
def load_parquet(filepath: str) -> pl.DataFrame:
    return pl.read_parquet(filepath).with_columns(
        pl.col("Category_Name").replace(CATEGORY_MAPPING)
    )

df = load_parquet("nobel_gold_decade_trends.parquet")

fig = px.line(
    df,
    x="Decade",
    y="Decade_Total_Adjusted_Prize_SEK",
    color="Category_Name",
    markers=True
)

fig.update_layout(
    hovermode="x unified",
    xaxis=dict(title="Tiår", tickmode="linear",dtick=10),
    yaxis=dict(title="Totalt justert premiebeløp (SEK)"),
    legend_title_text="Kategori"
)

st.plotly_chart(fig, use_container_width=True)