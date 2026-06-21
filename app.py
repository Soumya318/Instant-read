"""
Instant Read — Python edition (v2)
Upload a dataset, get instant profiling + SQL + ML + a plain-language Q&A panel.

Stack: Streamlit (UI), pandas/numpy (data), DuckDB (SQL), scikit-learn (ML),
Plotly (charts).

Run with:  streamlit run app.py
"""

import io
import re
import time

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================================================
# Page setup + theme
# ============================================================================
st.set_page_config(page_title="Instant Read", page_icon="◈", layout="wide")

PALETTE = ["#FF6B6B", "#4ECDC4", "#FFD166", "#A78BFA", "#5FCBD8", "#F2C14E"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 0%, rgba(255,107,107,0.07), transparent 40%),
                    radial-gradient(circle at 90% 10%, rgba(78,205,196,0.07), transparent 40%),
                    #0F1117;
        color: #F2F0E9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    [data-testid="stMetricValue"] { color: #FF6B6B; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 14px 16px;
    }
    .block-container { padding-top: 1.6rem; max-width: 1200px; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 2.6rem; line-height: 1.1; margin-bottom: 0;
        background: linear-gradient(90deg, #FF6B6B, #FFD166 40%, #4ECDC4 80%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #9aa0ad; font-size: 1.0rem; margin-top: 4px; margin-bottom: 1.4rem;}

    .badge {
        display:inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12.5px; font-weight: 600; margin: 3px 6px 3px 0;
    }
    .badge-teal { background: rgba(78,205,196,0.15); color:#4ECDC4; }
    .badge-coral { background: rgba(255,107,107,0.15); color:#FF6B6B; }
    .badge-gold { background: rgba(255,209,102,0.15); color:#FFD166; }
    .badge-purple { background: rgba(167,139,250,0.15); color:#A78BFA; }

    .story-card {
        background: linear-gradient(135deg, rgba(255,107,107,0.06), rgba(78,205,196,0.06));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        padding: 22px 26px; font-size: 15.5px; line-height: 1.7;
    }

    .qa-answer {
        background: rgba(78,205,196,0.08); border-left: 3px solid #4ECDC4;
        border-radius: 8px; padding: 14px 18px; margin-top: 10px; font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">◈ Instant Read</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a dataset → profiling, SQL, ML, and a Q&A panel — all instant.</div>',
    unsafe_allow_html=True,
)

TOOLS_USED = ["Streamlit", "pandas", "NumPy", "DuckDB (SQL)", "scikit-learn (ML)", "Plotly"]
st.markdown(
    "".join(f'<span class="badge badge-teal">{t}</span>' for t in TOOLS_USED),
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# Upload
# ============================================================================
uploaded = st.file_uploader("Drop a CSV or Excel file", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Waiting for a file. Try any CSV/XLSX you have — sales data, survey exports, logs, etc.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


# Quick "scanning" feel — completes in under a second, mirrors the file actually being read.
scan_box = st.empty()
scan_steps = ["reading bytes…", "inferring column types…", "computing distributions…", "done."]
progress = scan_box.progress(0, text=scan_steps[0])
for i, step in enumerate(scan_steps):
    progress.progress(int((i + 1) / len(scan_steps) * 100), text=step)
    time.sleep(0.12)
scan_box.empty()

df = load_data(uploaded.getvalue(), uploaded.name)
st.toast("Dataset analyzed!", icon="✅")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]
n_rows, n_cols = df.shape

# ============================================================================
# Health score + data story (computed once, used in multiple tabs)
# ============================================================================
def compute_health_score(df, numeric_cols):
    missing_pct = df.isna().mean().mean() * 100
    dup_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) else 0

    outlier_pct = 0
    if numeric_cols:
        outlier_counts = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
            outlier_counts.append(out / len(s) * 100)
        outlier_pct = np.mean(outlier_counts) if outlier_counts else 0

    score = 100 - (missing_pct * 0.5) - (dup_pct * 0.8) - (outlier_pct * 0.3)
    return max(0, min(100, round(score))), round(missing_pct, 1), round(dup_pct, 1), round(outlier_pct, 1)


health_score, missing_pct, dup_pct, outlier_pct = compute_health_score(df, numeric_cols)


def health_badge(score):
    if score >= 85:
        return "Excellent ✅", "#4ECDC4"
    if score >= 65:
        return "Decent ⚠️", "#FFD166"
    return "Needs cleaning 🔧", "#FF6B6B"


badge_text, badge_color = health_badge(health_score)


def build_data_story(df, numeric_cols, categorical_cols, n_rows, n_cols):
    lines = [f"This dataset has **{n_rows:,} rows** and **{n_cols} columns** "
             f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)."]

    if missing_pct > 0:
        worst_col = df.isna().mean().idxmax()
        worst_pct = round(df[worst_col].isna().mean() * 100, 1)
        lines.append(f"Roughly **{missing_pct}%** of all cells are missing — the gappiest column is "
                      f"**`{worst_col}`** at {worst_pct}% missing.")
    else:
        lines.append("There are **no missing values** anywhere in the dataset.")

    if dup_pct > 0:
        lines.append(f"About **{dup_pct}%** of rows look like duplicates of another row.")

    if numeric_cols:
        spread_col = df[numeric_cols].std().idxmax()
        lines.append(f"Among numeric columns, **`{spread_col}`** has the widest spread of values "
                      f"(highest standard deviation).")

    if categorical_cols:
        cat = categorical_cols[0]
        top_val = df[cat].astype(str).value_counts().idxmax()
        top_share = round(df[cat].astype(str).value_counts(normalize=True).max() * 100, 1)
        lines.append(f"In **`{cat}`**, the most common value is **\"{top_val}\"**, "
                      f"covering {top_share}% of rows.")

    return " ".join(lines)


data_story = build_data_story(df, numeric_cols, categorical_cols, n_rows, n_cols)

# ============================================================================
# Tabs
# ============================================================================
tab_overview, tab_story, tab_profile, tab_ask, tab_sql, tab_ml, tab_export = st.tabs(
    ["Overview", "Data Story & Health", "Profiling", "Ask Your Data", "SQL Explorer", "ML Insights", "Export / Power BI"]
)

# ---------------- Overview ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", n_cols)
    c3.metric("Numeric columns", len(numeric_cols))
    c4.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Column types")
    dtype_table = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum().values,
        "missing %": (df.isna().mean() * 100).round(1).values,
        "unique values": df.nunique().values,
    })
    st.dataframe(dtype_table, width="stretch")

# ---------------- Data Story & Health ----------------
with tab_story:
    colA, colB = st.columns([1.4, 1])

    with colA:
        st.subheader("📖 Auto-generated summary")
        st.markdown(f'<div class="story-card">{data_story}</div>', unsafe_allow_html=True)

    with colB:
        st.subheader("💚 Data Health Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": " / 100", "font": {"color": "#F2F0E9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aa0ad"},
                "bar": {"color": badge_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 65], "color": "rgba(255,107,107,0.15)"},
                    {"range": [65, 85], "color": "rgba(255,209,102,0.15)"},
                    {"range": [85, 100], "color": "rgba(78,205,196,0.15)"},
                ],
            },
        ))
        fig.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F2F0E9"})
        st.plotly_chart(fig, width="stretch")
        st.markdown(f'<span class="badge badge-purple">{badge_text}</span>', unsafe_allow_html=True)
        st.caption(f"Missing: {missing_pct}% · Duplicate rows: {dup_pct}% · Outlier-heavy numeric cells: {outlier_pct}%")

# ---------------- Profiling ----------------
with tab_profile:
    if numeric_cols:
        st.subheader("Numeric distributions")
        sel_num = st.multiselect("Columns to plot", numeric_cols, default=numeric_cols[:4])
        cols = st.columns(2)
        for i, col in enumerate(sel_num):
            fig = px.histogram(df, x=col, nbins=30, title=col,
                                color_discrete_sequence=[PALETTE[i % len(PALETTE)]])
            fig.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            cols[i % 2].plotly_chart(fig, width="stretch")

        st.subheader("Summary statistics")
        st.dataframe(df[numeric_cols].describe().T, width="stretch")

        if len(numeric_cols) >= 2:
            st.subheader("Correlation matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
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
                         color_discrete_sequence=[PALETTE[(i + 2) % len(PALETTE)]])
            fig.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(categoryorder="total ascending"))
            cols[i % 2].plotly_chart(fig, width="stretch")

# ---------------- Ask Your Data (Q&A) ----------------
with tab_ask:
    st.subheader("💬 Ask Your Data")
    st.caption("Type a question in plain English, or tap one of the quick questions below.")

    def answer_question(q, df, numeric_cols, categorical_cols):
        ql = q.lower().strip()
        cols_lower = {c.lower(): c for c in df.columns}

        def find_col():
            for cl, orig in cols_lower.items():
                if cl in ql:
                    return orig
            return None

        if re.search(r"how many rows|number of rows|row count", ql):
            return f"This dataset has **{n_rows:,} rows**."
        if re.search(r"how many columns|number of columns|column count", ql):
            return f"This dataset has **{n_cols} columns**."
        if re.search(r"tool|library|stack|built with|technology", ql):
            return "Built with: " + ", ".join(TOOLS_USED) + "."
        if re.search(r"missing", ql):
            col = find_col()
            if col:
                m = df[col].isna().sum()
                return f"**`{col}`** has **{m}** missing values ({round(m/n_rows*100,1)}%)."
            return f"Total missing values across the dataset: **{int(df.isna().sum().sum())}** ({missing_pct}% of all cells)."
        if re.search(r"duplicate", ql):
            return f"There are **{df.duplicated().sum()}** duplicate rows ({dup_pct}%)."
        if re.search(r"health|quality|clean", ql):
            return f"Data Health Score: **{health_score}/100** — {badge_text}."
        if re.search(r"numeric column|number column", ql):
            return f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}."
        if re.search(r"categorical column|text column", ql):
            return f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}."
        if re.search(r"list.*column|what columns|column names", ql):
            return f"Columns: {', '.join(df.columns)}."
        if re.search(r"average|mean", ql):
            col = find_col()
            if col and col in numeric_cols:
                return f"The average of **`{col}`** is **{df[col].mean():.2f}**."
            return "Tell me which numeric column you'd like the average of — mention its name in the question."
        if re.search(r"max|highest|largest", ql):
            col = find_col()
            if col and col in numeric_cols:
                return f"The maximum value of **`{col}`** is **{df[col].max():.2f}**."
        if re.search(r"min|lowest|smallest", ql):
            col = find_col()
            if col and col in numeric_cols:
                return f"The minimum value of **`{col}`** is **{df[col].min():.2f}**."
        if re.search(r"unique|distinct", ql):
            col = find_col()
            if col:
                return f"**`{col}`** has **{df[col].nunique()}** unique values."
        if re.search(r"most common|most frequent|top value|mode", ql):
            col = find_col()
            if col:
                v = df[col].astype(str).value_counts().idxmax()
                return f"The most common value in **`{col}`** is **\"{v}\"**."
        if re.search(r"correlat", ql):
            mentioned = [c for c in numeric_cols if c.lower() in ql]
            if len(mentioned) >= 2:
                r = df[mentioned[0]].corr(df[mentioned[1]])
                return f"Correlation between **`{mentioned[0]}`** and **`{mentioned[1]}`** is **{r:.2f}**."
            return "Mention two numeric column names to get their correlation."
        if re.search(r"outlier", ql):
            return f"About **{outlier_pct}%** of numeric values look like outliers (IQR method)."
        return ("I couldn't quite match that to a stat I track yet. Try asking about rows, columns, "
                "missing values, duplicates, averages, max/min, unique values, most common value, "
                "correlation between two columns, or data health.")

    quick_qs = [
        "How many rows and columns?",
        "How many missing values?",
        "What tools were used?",
        "What's the data health score?",
        "List all columns",
    ]
    qcols = st.columns(len(quick_qs))
    clicked = None
    for i, qtext in enumerate(quick_qs):
        if qcols[i].button(qtext, key=f"qbtn_{i}"):
            clicked = qtext

    user_q = st.text_input("Or type your own question", value=clicked or "", placeholder="e.g. What's the average of price?")
    if user_q:
        ans = answer_question(user_q, df, numeric_cols, categorical_cols)
        st.markdown(f'<div class="qa-answer">{ans}</div>', unsafe_allow_html=True)

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
            st.dataframe(result, width="stretch")
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
                              color_discrete_map={"normal": "#4ECDC4", "outlier": "#FF6B6B"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            st.caption("Isolation Forest flags points that are easy to isolate from the rest.")
            st.dataframe(work_df[work_df["is_outlier"] == "outlier"].head(50), width="stretch")

        elif ml_section == "Clustering":
            k = st.slider("Number of clusters (k)", 2, 10, 3)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(scaled)

            pca = PCA(n_components=2)
            proj = pca.fit_transform(scaled)
            plot_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
            plot_df["cluster"] = labels.astype(str)
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster",
                              color_discrete_sequence=PALETTE)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

            sizes = pd.Series(labels).value_counts().sort_index().rename("rows").reset_index()
            sizes.columns = ["cluster", "rows"]
            st.dataframe(sizes, width="stretch")

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
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
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

                importances = pd.DataFrame({
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False).head(15)
                fig = px.bar(importances, x="importance", y="feature", orientation="h",
                             color_discrete_sequence=["#FFD166"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   yaxis=dict(categoryorder="total ascending"))
                st.plotly_chart(fig, width="stretch")

# ---------------- Export / Power BI ----------------
with tab_export:
    st.subheader("Export")
    st.download_button(
        "Download dataset as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv",
    )

    report_lines = [
        f"Rows: {n_rows}",
        f"Columns: {n_cols}",
        f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}",
        f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}",
        f"Data Health Score: {health_score}/100 ({badge_text})",
        "",
        "Data story:",
        data_story.replace("**", ""),
        "",
        "Summary statistics:",
        df.describe(include="all").to_string(),
    ]
    st.download_button(
        "Download profiling report (.txt)",
        "\n".join(report_lines).encode("utf-8"),
        file_name="profiling_report.txt",
        mime="text/plain",
    )

    st.subheader("Connect this data to Power BI")
    st.markdown(
        """
        This app handles fast, ad-hoc profiling and ML. For a shareable, scheduled
        dashboard, point Power BI at the same data:

        1. **Simplest path** — download the CSV above, then in Power BI Desktop:
           `Get Data → Text/CSV` and pick the file.
        2. **Live database path** — load your data into a SQL database (Postgres,
           SQL Server, etc.). Both this app and Power BI can then query the same source.
        3. **Python in Power BI** — Power BI Desktop's Python script visual can run
           the same pandas/scikit-learn code shown here, directly inside a report.
        """
    )
