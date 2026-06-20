# Instant Read — Python edition

A dataset-profiling app built with real data-analytics tooling: **Python, pandas,
SQL (DuckDB), scikit-learn (ML), and Streamlit** for the UI — with a path to
**Power BI** for dashboards.

## What it does

Upload a CSV or Excel file and instantly get:

- **Overview** — shape, dtypes, missing values, preview
- **Profiling** — histograms, summary statistics, categorical breakdowns, correlation matrix
- **SQL Explorer** — query the uploaded data directly with SQL (via DuckDB, no database setup needed)
- **ML Insights**
  - Outlier detection (Isolation Forest)
  - Clustering (KMeans + PCA visualization)
  - A quick baseline predictive model (Random Forest) on any column you pick as the target, auto-detecting classification vs. regression
- **Export** — download the data or a profiling report, plus instructions for wiring the same data into Power BI

## Stack

| Layer | Tool |
|---|---|
| UI / app framework | Streamlit |
| Data handling | pandas, numpy |
| SQL queries on the file | DuckDB |
| Charts | Plotly |
| Machine learning | scikit-learn (IsolationForest, KMeans, PCA, RandomForest) |
| Dashboards (downstream) | Power BI, via CSV or a shared SQL database |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) and
drop in a CSV or Excel file.

## Connecting to Power BI

This app is built for fast, ad-hoc exploration. For a scheduled or shared
dashboard, point Power BI at the same data:

1. **CSV path** — use the "Export" tab to download a CSV, then in Power BI:
   `Get Data → Text/CSV`.
2. **Shared database path** — load your data into a SQL database (Postgres,
   SQL Server, etc.). Power BI connects via `Get Data → Database`, and this
   app can be pointed at the same database (swap the DuckDB `register` call
   for a SQLAlchemy connection) so both tools read the same numbers.
3. **Python visual in Power BI** — Power BI Desktop supports a Python script
   visual, so the ML views here (outlier detection, clustering) can also be
   embedded directly inside a Power BI report if you'd rather keep everything
   in one place.

## Notes

- Everything runs locally — no data leaves your machine.
- The predictive model tab is a baseline (untuned Random Forest) meant to
  surface which features matter, not a production model.
