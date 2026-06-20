"""
Instant Read — Python edition
A Streamlit app that profiles any uploaded dataset instantly:
- Pandas for data handling
- DuckDB for SQL queries directly on the uploaded file
- Plotly for charts
- scikit-learn for ML (outlier detection, clustering, quick baseline model)

Run with:  streamlit run app.py
"""

import io
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Instant Read", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14171F; color: #F2F0E9; }
    [data-testid="stMetricValue"] { color: #E8A23D; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("◈ Instant Read")
st.caption("Upload a dataset → get profiling, SQL access, and ML insights immediately.")

# ----------------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------------
uploaded = st.file_uploader("Drop a CSV or Excel file", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Waiting for a file. Try any CSV/XLSX you have — sales data, survey exports, logs, etc.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


df = load_data(uploaded.getvalue(), uploaded.name)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_profile, tab_sql, tab_ml, tab_export = st.tabs(
    ["Overview", "Profiling", "SQL Explorer", "ML Insights", "Export / Power BI"]
)

# ---------------- Overview ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric columns", len(numeric_cols))
    c4.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column types")
    dtype_table = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum().values,
        "missing %": (df.isna().mean() * 100).round(1).values,
        "unique values": df.nunique().values,
    })
    st.dataframe(dtype_table, use_container_width=True)

# ---------------- Profiling ----------------
with tab_profile:
    if numeric_cols:
        st.subheader("Numeric distributions")
        sel_num = st.multiselect("Columns to plot", numeric_cols, default=numeric_cols[:4])
        cols = st.columns(2)
        for i, col in enumerate(sel_num):
            fig = px.histogram(df, x=col, nbins=30, title=col,
                                color_discrete_sequence=["#E8A23D"])
            fig.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            cols[i % 2].plotly_chart(fig, use_container_width=True)

        st.subheader("Summary statistics")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("Correlation matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    if categorical_cols:
        st.subheader("Categorical breakdowns")
        sel_cat = st.multiselect("Columns to inspect", categorical_cols, default=categorical_cols[:3])
        cols = st.columns(2)
        for i, col in enumerate(sel_cat):
            top = df[col].astype(str).value_counts().head(10).reset_index()
            top.columns = [col, "count"]
            fig = px.bar(top, x="count", y=col, orientation="h",
                         color_discrete_sequence=["#5FCBD8"])
            fig.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(categoryorder="total ascending"))
            cols[i % 2].plotly_chart(fig, use_container_width=True)

# ---------------- SQL Explorer (DuckDB) ----------------
with tab_sql:
    st.subheader("Query this dataset with SQL")
    st.caption("The uploaded file is registered as a table called `data`. DuckDB runs the query in-memory.")

    default_query = "SELECT * FROM data LIMIT 10;"
    query = st.text_area("SQL query", value=default_query, height=120)

    if st.button("Run query"):
        try:
            con = duckdb.connect()
            con.register("data", df)
            result = con.execute(query).fetchdf()
            st.success(f"{len(result):,} rows returned")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download result as CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Query failed: {e}")

    with st.expander("Example queries"):
        st.code(
            "-- Top categories\n"
            f"SELECT {categorical_cols[0] if categorical_cols else 'col'}, COUNT(*) AS n "
            "FROM data GROUP BY 1 ORDER BY n DESC LIMIT 10;\n\n"
            "-- Numeric summary by group\n"
            f"SELECT {categorical_cols[0] if categorical_cols else 'col'}, "
            f"AVG({numeric_cols[0] if numeric_cols else 'val'}) AS avg_val "
            "FROM data GROUP BY 1 ORDER BY avg_val DESC;",
            language="sql",
        )

# ---------------- ML Insights ----------------
with tab_ml:
    st.subheader("Machine learning insights")

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for the ML views below.")
    else:
        ml_section = st.radio(
            "Choose an analysis",
            ["Outlier detection", "Clustering", "Quick predictive model"],
            horizontal=True,
        )

        work_df = df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(work_df)

        if ml_section == "Outlier detection":
            contamination = st.slider("Expected outlier fraction", 0.01, 0.25, 0.05, 0.01)
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(scaled)
            work_df = work_df.copy()
            work_df["is_outlier"] = np.where(preds == -1, "outlier", "normal")

            n_out = (work_df["is_outlier"] == "outlier").sum()
            st.metric("Outliers flagged", f"{n_out} / {len(work_df)}")

            pca = PCA(n_components=2)
            proj = pca.fit_transform(scaled)
            plot_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
            plot_df["is_outlier"] = work_df["is_outlier"].values
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="is_outlier",
                              color_discrete_map={"normal": "#5FCBD8", "outlier": "#E36464"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Isolation Forest flags points that are easy to isolate from the rest — a common way to surface anomalies without labeled data.")
            st.dataframe(work_df[work_df["is_outlier"] == "outlier"].head(50), use_container_width=True)

        elif ml_section == "Clustering":
            k = st.slider("Number of clusters (k)", 2, 10, 3)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(scaled)

            pca = PCA(n_components=2)
            proj = pca.fit_transform(scaled)
            plot_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
            plot_df["cluster"] = labels.astype(str)
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster",
                              color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("KMeans on standardized numeric columns, visualized in 2D via PCA.")

            sizes = pd.Series(labels).value_counts().sort_index().rename("rows").reset_index()
            sizes.columns = ["cluster", "rows"]
            st.dataframe(sizes, use_container_width=True)

        else:  # Quick predictive model
            target = st.selectbox("Target column to predict", df.columns.tolist())
            features = [c for c in df.columns if c != target]
            model_df = df[[target] + features].dropna()

            if model_df[target].nunique() <= 15 and (model_df[target].dtype == object or model_df[target].nunique() < 15):
                task = "classification"
            else:
                task = "regression"
            st.write(f"Detected task type: **{task}**")

            X = pd.get_dummies(model_df[features], drop_first=True)
            y = model_df[target]

            if task == "classification" and y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y)

            if len(X) < 20:
                st.warning("Not enough rows after dropping missing values to train a reliable model.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
                if task == "classification":
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    c1, c2 = st.columns(2)
                    c1.metric("Accuracy", f"{acc:.2%}")
                    c2.metric("F1 (weighted)", f"{f1:.2f}")
                else:
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = mean_squared_error(y_test, preds) ** 0.5
                    r2 = r2_score(y_test, preds)
                    c1, c2 = st.columns(2)
                    c1.metric("RMSE", f"{rmse:.3f}")
                    c2.metric("R²", f"{r2:.3f}")

                st.caption("Baseline Random Forest — a quick read on which features matter, not a tuned production model.")
                importances = pd.DataFrame({
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False).head(15)
                fig = px.bar(importances, x="importance", y="feature", orientation="h",
                             color_discrete_sequence=["#E8A23D"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   yaxis=dict(categoryorder="total ascending"))
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Export / Power BI ----------------
with tab_export:
    st.subheader("Export")
    st.download_button(
        "Download dataset as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv",
    )

    summary_lines = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}",
        f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}",
        "",
        "Summary statistics:",
        df.describe(include="all").to_string(),
    ]
    st.download_button(
        "Download profiling report (.txt)",
        "\n".join(summary_lines).encode("utf-8"),
        file_name="profiling_report.txt",
        mime="text/plain",
    )

    st.subheader("Connect this data to Power BI")
    st.markdown(
        """
        This app handles fast, ad-hoc profiling and ML. For a shareable, scheduled
        dashboard, point Power BI at the same data:

        1. **Simplest path** — download the CSV above, then in Power BI Desktop:
           `Get Data → Text/CSV` and pick the file. Build visuals on top as usual.
        2. **Live database path** — instead of uploading files here, load your data into
           a SQL database (Postgres, SQL Server, etc.). Both this app (via DuckDB/SQLAlchemy)
           and Power BI (`Get Data → Database`) can then query the same source, so the
           numbers always match.
        3. **Python in Power BI** — Power BI Desktop has a *Python script* visual, which
           can run pandas/scikit-learn code directly inside a report if you want the ML
           views embedded there instead of in this app.
        """
    )
