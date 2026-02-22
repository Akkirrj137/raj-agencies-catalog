import os
import re
from io import BytesIO
from textwrap import shorten
from typing import Optional, List

import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ================= CONFIG =================
st.set_page_config(page_title="RAJ GROUP • Catalog", layout="wide")

MAX_CARDS = 240
DEFAULT_DATA_PATH = os.path.join("data", "master.xlsx")
LOGO_PATH = os.path.join("assets", "logo.png")

# ================= HELPERS =================
@st.cache_data(show_spinner=False)
def read_sheet(path_or_file, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path_or_file, sheet_name=sheet_name)

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None

def df_contains_search(df: pd.DataFrame, cols: List[Optional[str]], q: str) -> pd.DataFrame:
    q = (q or "").lower().strip()
    if not q:
        return df
    mask = False
    for c in cols:
        if c and c in df.columns:
            mask = mask | safe_str_series(df[c]).str.lower().str.contains(q, na=False)
    return df[mask]

# ================= LOAD DATA (PUBLIC MODE) =================
with st.sidebar:
    st.header("Data")

uploaded = st.sidebar.file_uploader(
    "Upload Excel (optional)",
    type=["xlsx", "xlsm"]
)

if uploaded is not None:
    xl = pd.ExcelFile(uploaded)
    sheet = st.sidebar.selectbox("Sheet", xl.sheet_names)
    df = read_sheet(uploaded, sheet)
else:
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error("❌ data/master.xlsx missing. Please upload or add file.")
        st.stop()
    xl = pd.ExcelFile(DEFAULT_DATA_PATH)
    sheet = xl.sheet_names[0]
    df = read_sheet(DEFAULT_DATA_PATH, sheet)

df.columns = [str(c).strip() for c in df.columns]

# ================= DETECT COLUMNS =================
col_code = pick_col(df, ["code", "part no", "part_no"])
col_desc = pick_col(df, ["description", "item name"])
col_mrp = pick_col(df, ["mrp", "price"])
col_rate = pick_col(df, ["rate"])
col_unit = pick_col(df, ["unit"])
col_group = pick_col(df, ["group"])
col_hsn = pick_col(df, ["hsn"])
col_gst = pick_col(df, ["gst"])
col_category = pick_col(df, ["category"])

# ================= HEADER =================
c1, c2 = st.columns([1,4])

with c1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH)

with c2:
    st.title("RAJ GROUP • Catalog")
    st.caption("Public searchable product catalog")

# ================= FILTERS =================
st.subheader("Filters")

f1, f2, f3 = st.columns(3)

selected_group = None
if col_group:
    groups = sorted(df[col_group].dropna().unique().tolist())
    with f1:
        selected_group = st.selectbox("Group", groups)

with f2:
    search_q = st.text_input("Search (Code / Description)")

with f3:
    mobile_mode = st.toggle("📱 Mobile compact", value=True)

# ================= APPLY FILTERS =================
out = df.copy()

if selected_group and col_group:
    out = out[safe_str_series(out[col_group]) == str(selected_group)]

out = df_contains_search(out, [col_code, col_desc], search_q)

# ================= KPI =================
k1, k2 = st.columns(2)
k1.metric("Total Rows", f"{len(df):,}")
k2.metric("Filtered Rows", f"{len(out):,}")

# ================= PDF EXPORT =================
st.subheader("Print / Export")

def build_a4_pdf(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("RAJ GROUP Catalog", styles["Title"]), Spacer(1,10)]

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.red),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()

pdf_bytes = build_a4_pdf(out.head(120))
st.download_button("⬇️ Download A4 PDF", data=pdf_bytes, file_name="catalog.pdf")

# ================= PRODUCTS =================
st.subheader("Products")

N = min(len(out), MAX_CARDS)
data = out.head(N)

grid_cols = st.columns(2 if mobile_mode else 3)

def badge(txt):
    return f"<span style='border:1px solid #888;padding:3px 8px;border-radius:999px;margin-right:6px'>{txt}</span>"

for i, (_, r) in enumerate(data.iterrows()):
    with grid_cols[i % len(grid_cols)]:
        st.markdown("---")

        code = str(r[col_code]) if col_code else ""
        desc = str(r[col_desc]) if col_desc else ""
        rate = str(r[col_rate]) if col_rate else ""
        mrp = str(r[col_mrp]) if col_mrp else ""
        unit = str(r[col_unit]) if col_unit else ""
        hsn = str(r[col_hsn]) if col_hsn else ""
        gst = str(r[col_gst]) if col_gst else ""

        st.markdown(f"**{code}**")
        st.write(shorten(desc, width=120))

        badges = []
        if rate and rate != "nan":
            badges.append(badge(f"RATE: {rate}"))
        if mrp and mrp != "nan":
            badges.append(badge(f"MRP: {mrp}"))
        if unit and unit != "nan":
            badges.append(badge(f"UNIT: {unit}"))
        if hsn and hsn != "nan":
            badges.append(badge(f"HSN: {hsn}"))
        if gst and gst != "nan":
            badges.append(badge(f"GST: {gst}"))

        st.markdown(" ".join(badges), unsafe_allow_html=True)

st.caption(f"Showing {N} products")