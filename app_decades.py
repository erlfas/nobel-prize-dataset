import plotly.express as px
import polars as pl
import streamlit as st

st.title("Decades")

@st.cache_data
def load_parquet(filepath: str) -> pl.DataFrame:
    return pl.read_parquet(filepath)

df = load_parquet("nobel_gold_decade_trends.parquet")

fig = px.line(
    df,
    x="Decade",
    y="Decade_Total_Adjusted_Prize_SEK",
    color="Category_Name",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)