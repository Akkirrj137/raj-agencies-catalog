import os
import re
import base64
from io import BytesIO
from textwrap import shorten
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st

from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="RAJ GROUP • Catalog", layout="wide")

# ==================== CONFIG ====================
MAX_CARDS = 240
REPO_IMG_DIR = os.path.join("assets", "products")
LOGO_PATH = os.path.join("assets", "logo.png")

# Dark Royal Blue theme
C_BG = "#071a3a"         # royal dark
C_BG2 = "#06132c"
C_CARD = "rgba(255,255,255,0.05)"
C_BORDER = "rgba(255,255,255,0.14)"
C_TEXT = "rgba(255,255,255,0.92)"

# Accent (logo-like)
C_RED = "#d81b28"
C_BLUE = "#1f6fb2"
C_YELLOW = "#f2b705"
C_ORANGE = "#f59e0b"

WATERMARK_OPACITY = 0.15  # 15% transparency


# ==================== HELPERS ====================
@st.cache_data(show_spinner=False)
def read_sheet(uploaded_file, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols:
            return cols[key]
    return None

def normalize_spaces(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())

def normalize_segment_std(val: str) -> str:
    s = normalize_spaces(val).upper()
    if "HCV" in s:
        return "HCV"
    if "LCV" in s:
        return "LCV"
    if "CAR" in s:
        return "CAR"
    return "OTHER"

def normalize_top_type(val: str) -> str:
    s = normalize_spaces(val).upper()
    if "2W" in s or "2 W" in s or "TWO WHEEL" in s or "2WHEEL" in s:
        return "2W"
    if "3W" in s or "3 W" in s or "THREE WHEEL" in s or "3WHEEL" in s:
        return "3W"
    if "EARTH" in s or "JCB" in s or "EXCAV" in s or "LOADER" in s or "BACKHOE" in s:
        return "EARTHMOVERS"
    if "TRACTOR" in s or "MF" in s or "MASSEY" in s:
        return "TRACTOR"
    if "HCV" in s:
        return "HCV"
    if "LCV" in s:
        return "LCV"
    if "CAR" in s:
        return "CAR"
    if "UNIVERSAL" in s or "GENERIC" in s:
        return "UNIVERSAL"
    return "OTHER"

def df_contains_search(df: pd.DataFrame, cols: List[Optional[str]], q: str) -> pd.DataFrame:
    q = (q or "").strip().lower()
    if not q:
        return df
    mask = False
    for c in cols:
        if c and c in df.columns:
            mask = mask | safe_str_series(df[c]).str.lower().str.contains(q, na=False)
    return df[mask]

def read_logo_bytes() -> Optional[bytes]:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return f.read()
    return None

def get_repo_image_bytes(code: str) -> Optional[bytes]:
    if not code or not os.path.isdir(REPO_IMG_DIR):
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(REPO_IMG_DIR, f"{code}{ext}")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    return None

@st.cache_data(show_spinner=False)
def watermark_image_bytes(img_bytes: bytes, logo_bytes: bytes, opacity: float = 0.15) -> bytes:
    """
    Add watermark (logo) on bottom-right with given opacity.
    Returns PNG bytes.
    """
    base = Image.open(BytesIO(img_bytes)).convert("RGBA")
    logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")

    # resize logo relative to image
    bw, bh = base.size
    target_w = max(120, int(bw * 0.22))
    ratio = target_w / logo.size[0]
    target_h = int(logo.size[1] * ratio)
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    # apply opacity
    alpha = logo.split()[-1]
    alpha = alpha.point(lambda p: int(p * opacity))
    logo.putalpha(alpha)

    # position (bottom-right)
    margin = max(10, int(min(bw, bh) * 0.03))
    x = bw - logo.size[0] - margin
    y = bh - logo.size[1] - margin

    out = Image.new("RGBA", base.size, (0, 0, 0, 0))
    out.paste(base, (0, 0))
    out.paste(logo, (x, y), logo)

    buf = BytesIO()
    out.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def build_a4_pdf(df: pd.DataFrame, title: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18, rightMargin=18,
        topMargin=18, bottomMargin=18,
    )
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 10)]

    if df.empty:
        story.append(Paragraph("No data to print.", styles["Normal"]))
        doc.build(story)
        return buf.getvalue()

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_RED)),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


# ==================== GLOBAL CSS (ROYAL BLUE + ZOOM EFFECT) ====================
st.markdown(
    f"""
    <style>
      /* App background */
      .stApp {{
        background: radial-gradient(1200px 800px at 15% 10%, {C_BG} 0%, {C_BG2} 50%, #040b1a 100%) !important;
        color: {C_TEXT};
      }}
      /* Remove default top padding a bit */
      .block-container {{
        padding-top: 1.2rem;
      }}

      /* Banner */
      .banner {{
        background: linear-gradient(90deg, rgba(31,111,178,0.95), rgba(216,27,40,0.95));
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
      }}
      .banner h1 {{
        color:#fff; margin:0; font-weight:900; letter-spacing:.3px; font-size:28px;
      }}
      .banner p {{
        color:rgba(255,255,255,.9); margin:4px 0 0 0; font-size:13px;
      }}
      .pill {{
        background: rgba(255,255,255,0.12);
        color: white;
        border: 1px solid rgba(255,255,255,0.20);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        white-space:nowrap;
      }}

      /* Filters box */
      .filtersbox {{
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 12px 12px;
        background: rgba(255,255,255,0.04);
        margin-bottom: 12px;
      }}

      /* Card */
      .card {{
        border: 1px solid {C_BORDER};
        border-radius: 18px;
        padding: 12px;
        background: {C_CARD};
        margin-bottom: 12px;
        box-shadow: 0 14px 40px rgba(0,0,0,0.25);
      }}
      .card:hover {{
        border-color: rgba(242,183,5,0.55);
      }}
      .code {{
        font-size: 16px; font-weight: 900; color: #fff; letter-spacing: .2px;
      }}

      /* Effective zoom effect on images inside card */
      .card img {{
        border-radius: 14px !important;
        transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
      }}
      .card img:hover {{
        transform: scale(1.08) translateY(-2px);
        box-shadow: 0 18px 46px rgba(0,0,0,0.40);
        filter: saturate(1.08) contrast(1.03);
      }}

      .badge {{
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,.18);
        font-size:12px;
        margin-right:6px;
        margin-top:6px;
        color: rgba(255,255,255,0.92);
      }}
      .bYellow {{ border-color: rgba(242,183,5,0.60); }}
      .bBlue {{ border-color: rgba(31,111,178,0.55); }}
      .bRed {{ border-color: rgba(216,27,40,0.55); }}
      .bDim {{ border-color: rgba(255,255,255,0.18); }}

      /* Buttons */
      div.stButton > button {{
        border-radius: 14px !important;
        font-weight: 900 !important;
        padding: .55rem .85rem !important;
      }}

      /* Inputs look nicer on dark */
      input, textarea {{
        color: #fff !important;
      }}
      label, .stMarkdown, .stText, .stCaption {{
        color: rgba(255,255,255,0.88) !important;
      }}

      @media (max-width: 768px) {{
        .banner h1 {{ font-size: 20px; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Excel (.xlsx/.xlsm)", type=["xlsx", "xlsm"])
    st.caption("Tip: Excel me naye columns add karoge to app auto-detect karega.")
    st.divider()

    st.header("Images (optional)")
    st.caption("Filename = Code (AA1001.jpg). Upload once per session.")
    imgs = st.file_uploader("Upload product images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if not uploaded:
    st.info("Upload your Excel file to start.")
    st.stop()

xl = pd.ExcelFile(uploaded)
with st.sidebar:
    sheet = st.selectbox("Sheet", xl.sheet_names, index=0)

df = read_sheet(uploaded, sheet)
df.columns = [str(c).strip() for c in df.columns]

# ==================== SESSION IMAGE MAP ====================
if "img_map" not in st.session_state:
    st.session_state.img_map = {}

if imgs:
    for f in imgs:
        code_key = os.path.splitext(f.name)[0].strip()
        st.session_state.img_map[code_key] = f.getvalue()

logo_bytes = read_logo_bytes()


# ==================== DETECT COLUMNS ====================
col_code = pick_col(df, ["code", "part no", "part_no", "partno"])
col_desc = pick_col(df, ["description", "item name", "item"])
col_mrp = pick_col(df, ["mrp", "list", "list price", "price", "rate"])
col_rate = pick_col(df, ["rate", "selling rate", "sale rate"])
col_unit = pick_col(df, ["unit", "uom"])
col_segment_raw = pick_col(df, ["segment"])
col_group = pick_col(df, ["group"])

# Extra details columns (auto detect)
col_hsn = pick_col(df, ["hsn", "hsn code", "hsn_code"])
col_gst = pick_col(df, ["gst", "gst %", "gst_percent", "gst rate"])
col_pack = pick_col(df, ["packing size", "packing", "pack size", "packing_size"])
col_std_pkg = pick_col(df, ["std pkg", "std package", "standard package", "std_pkg"])
col_variant = pick_col(df, ["variant", "varient"])
col_year = pick_col(df, ["year", "variant year", "model year"])
col_color = pick_col(df, ["color", "colour"])
col_oe = pick_col(df, ["oe-part no", "oe part no", "oe_part_no", "oem", "oem part no"])
col_coupon = pick_col(df, ["coupon", "coupan"])
col_category = pick_col(df, ["category name", "category"])

col_brand = pick_col(df, ["vehicle brand", "brand"])
col_vehicle = pick_col(df, ["vehicle"])
col_model = pick_col(df, ["model"])
col_img_url = pick_col(df, ["image_url", "img_url", "image link", "image_link", "photo_url"])


# Standard segment
df["SEGMENT_STD"] = df[col_segment_raw].astype(str).apply(normalize_segment_std) if col_segment_raw else "OTHER"
df["TYPE_TOP"] = df[col_segment_raw].astype(str).apply(normalize_top_type) if col_segment_raw else "OTHER"


# ==================== HEADER ====================
h1, h2 = st.columns([1.1, 4.2])
with h1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
with h2:
    st.markdown(
        f"""
        <div class="banner">
          <div>
            <h1>RAJ GROUP • Catalog</h1>
            <p>Royal Blue UI • Effective Zoom • Image Watermark • A4 Print</p>
          </div>
          <div class="pill">Rows: {len(df):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================== QUICK TOP FILTERS ====================
if "top_type" not in st.session_state:
    st.session_state.top_type = "ALL"

def top_btn(label: str, container):
    active = (st.session_state.top_type == label)
    klass = "topbtnActive" if active else "topbtn"
    with container:
        # Use native button but keep clean
        if st.button(label, use_container_width=True):
            st.session_state.top_type = label

row = st.columns([1,1,1,1,1,1,1,1,1])
top_btn("ALL", row[0])
top_btn("2W", row[1])
top_btn("3W", row[2])
top_btn("CAR", row[3])
top_btn("LCV", row[4])
top_btn("HCV", row[5])
top_btn("TRACTOR", row[6])
top_btn("EARTHMOVERS", row[7])
top_btn("UNIVERSAL", row[8])


# ==================== FILTERS (GROUP SEARCH MERGED) ====================
st.markdown('<div class="filtersbox">', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns([2.4, 1.4, 1.4, 1.2])

selected_group = None
with f1:
    if col_group:
        groups = sorted([x for x in safe_str_series(df[col_group]).unique().tolist() if str(x).strip() != ""])
        if groups:
            # IMPORTANT: No separate search input. Streamlit selectbox allows typing to search.
            selected_group = st.selectbox("Group (type here to search)", groups, index=0)
        else:
            st.warning("Group column present but empty.")
    else:
        st.warning("No GROUP column found in sheet.")

with f2:
    seg_opts = ["ALL"] + sorted([x for x in df["SEGMENT_STD"].unique().tolist() if x])
    seg_std = st.selectbox("Segment (deduped)", seg_opts, index=0)

with f3:
    mobile_mode = st.toggle("📱 Mobile compact", value=True)

with f4:
    q_search = st.text_input("Product Search (Code/Desc)", value="", placeholder="Type to filter cards...")

st.markdown("</div>", unsafe_allow_html=True)


# ==================== APPLY FILTERS ====================
out = df.copy()

# Mandatory group default (fast browsing)
if col_group and selected_group:
    out = out[safe_str_series(out[col_group]) == str(selected_group)]

# Top type filter
if st.session_state.top_type != "ALL":
    out = out[out["TYPE_TOP"] == st.session_state.top_type]

# Segment filter
if seg_std != "ALL":
    out = out[out["SEGMENT_STD"] == seg_std]

# Search (code/desc)
out = df_contains_search(out, [col_code, col_desc], q_search)

# KPI
k1, k2, k3 = st.columns(3)
k1.metric("Rows (total)", f"{len(df):,}")
k2.metric("Rows (filtered)", f"{len(out):,}")
k3.metric("Columns", f"{len(df.columns):,}")


# ==================== IMAGE RESOLUTION ====================
def resolve_image_url(row) -> Optional[str]:
    if col_img_url and col_img_url in row.index:
        url = str(row[col_img_url]).strip()
        if url.lower().startswith(("http://", "https://")):
            return url
    return None

def resolve_image_bytes_with_watermark(row) -> Optional[bytes]:
    """
    Priority:
    1) session uploaded image (code.jpg)
    2) repo assets/products/<code>.jpg
    Add watermark if logo exists.
    """
    code = str(row[col_code]).strip() if col_code else ""
    if not code:
        return None

    img_bytes = None
    if code in st.session_state.img_map:
        img_bytes = st.session_state.img_map[code]
    else:
        img_bytes = get_repo_image_bytes(code)

    if not img_bytes:
        return None

    if logo_bytes:
        try:
            return watermark_image_bytes(img_bytes, logo_bytes, opacity=WATERMARK_OPACITY)
        except Exception:
            return img_bytes
    return img_bytes


# ==================== PRINT / EXPORT ====================
st.subheader("Print / Export")

p1, p2, p3 = st.columns([2.2, 1.3, 1.5])

default_print_cols = []
for c in [
    col_code, col_desc, col_rate, col_mrp, col_unit,
    col_hsn, col_gst, col_pack, col_std_pkg, col_variant, col_year, col_color,
    col_oe, col_coupon, col_category, "SEGMENT_STD"
]:
    if c and c in out.columns:
        default_print_cols.append(c)
default_print_cols = default_print_cols[:8]  # A4 fit

with p1:
    print_cols = st.multiselect(
        "Print columns (A4 fit)",
        options=[c for c in out.columns.tolist()],
        default=default_print_cols
    )

with p2:
    print_rows = st.number_input("Max rows to print", min_value=10, max_value=500, value=120, step=10)

with p3:
    st.caption("A4 PDF download")
    title = f"RAJ GROUP • {selected_group or 'Group'} • {st.session_state.top_type}"
    pdf_bytes = build_a4_pdf(out[print_cols].head(int(print_rows)) if print_cols else out.head(int(print_rows)), title=title)
    st.download_button(
        "⬇️ Download A4 PDF",
        data=pdf_bytes,
        file_name="raj_catalog_print_a4.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ==================== PRODUCTS (CARDS) ====================
st.subheader("Products")

N = min(len(out), MAX_CARDS)
data = out.head(N)

grid_cols = st.columns(2 if mobile_mode else 3)

def badge(text: str, cls: str = "bDim") -> str:
    return f"<span class='badge {cls}'>{text}</span>"

for i, (_, r) in enumerate(data.iterrows()):
    with grid_cols[i % len(grid_cols)]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        code = str(r[col_code]).strip() if col_code else ""
        desc = str(r[col_desc]).strip() if col_desc else ""

        # values
        rate = str(r[col_rate]).strip() if col_rate else ""
        mrp = str(r[col_mrp]).strip() if col_mrp else ""
        unit = str(r[col_unit]).strip() if col_unit else ""
        hsn = str(r[col_hsn]).strip() if col_hsn else ""
        gst = str(r[col_gst]).strip() if col_gst else ""
        pack = str(r[col_pack]).strip() if col_pack else ""
        std_pkg = str(r[col_std_pkg]).strip() if col_std_pkg else ""
        variant = str(r[col_variant]).strip() if col_variant else ""
        year = str(r[col_year]).strip() if col_year else ""
        color = str(r[col_color]).strip() if col_color else ""
        oe = str(r[col_oe]).strip() if col_oe else ""
        coupon = str(r[col_coupon]).strip() if col_coupon else ""
        category = str(r[col_category]).strip() if col_category else ""

        seg = str(r.get("SEGMENT_STD", "")).strip()
        typ = str(r.get("TYPE_TOP", "")).strip()

        img_url = resolve_image_url(r)
        img_bytes = None if img_url else resolve_image_bytes_with_watermark(r)

        # Image
        if img_url:
            st.image(img_url, use_container_width=True)
        elif img_bytes:
            st.image(img_bytes, use_container_width=True)
        else:
            st.caption("No image")

        # Title
        st.markdown(f"<div class='code'>{code}</div>", unsafe_allow_html=True)
        st.write(shorten(desc, width=105 if mobile_mode else 160, placeholder="…"))

        # Badges row 1
        badges = []
        if rate and rate.lower() != "nan":
            badges.append(badge(f"RATE: {rate}", "bBlue"))
        if mrp and mrp.lower() != "nan":
            badges.append(badge(f"MRP: {mrp}", "bYellow"))
        if unit and unit.lower() != "nan":
            badges.append(badge(f"UNIT: {unit}", "bDim"))
        if typ and typ != "OTHER":
            badges.append(badge(typ, "bRed"))
        if seg:
            badges.append(badge(seg, "bDim"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

        # Details row (requested fields)
        detail_badges = []
        if hsn and hsn.lower() != "nan":
            detail_badges.append(badge(f"HSN: {hsn}", "bDim"))
        if gst and gst.lower() != "nan":
            detail_badges.append(badge(f"GST: {gst}", "bDim"))
        if pack and pack.lower() != "nan":
            detail_badges.append(badge(f"PACKING: {pack}", "bDim"))
        if std_pkg and std_pkg.lower() != "nan":
            detail_badges.append(badge(f"STD PKG: {std_pkg}", "bDim"))
        if variant and variant.lower() != "nan":
            detail_badges.append(badge(f"VARIANT: {variant}", "bDim"))
        if year and year.lower() != "nan":
            detail_badges.append(badge(f"YEAR: {year}", "bDim"))
        if color and color.lower() != "nan":
            detail_badges.append(badge(f"COLOR: {color}", "bDim"))
        if oe and oe.lower() != "nan":
            detail_badges.append(badge(f"OE: {oe}", "bDim"))
        if coupon and coupon.lower() != "nan":
            detail_badges.append(badge(f"COUPON: {coupon}", "bDim"))
        if category and category.lower() != "nan":
            detail_badges.append(badge(f"CATEGORY: {category}", "bBlue"))

        if detail_badges:
            st.markdown(" ".join(detail_badges), unsafe_allow_html=True)

        # Zoom (bigger view)
        with st.popover("🔍 Zoom image (big)"):
            if img_url:
                st.image(img_url, use_container_width=True)
            elif img_bytes:
                st.image(img_bytes, use_container_width=True)
            else:
                st.info("No image available for this product.")

        # Full row view
        with st.expander("View full row (all columns)"):
            st.dataframe(pd.DataFrame([r]), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.caption(f"Showing {N} cards (limit {MAX_CARDS}). Use filters to narrow results.")


# ==================== WHATSAPP EXPORT ====================
st.subheader("WhatsApp Share")

def build_whatsapp_text(df_: pd.DataFrame, max_rows=120) -> str:
    out_ = df_.head(max_rows)
    lines = []
    for _, rr in out_.iterrows():
        code = str(rr[col_code]).strip() if col_code else ""
        desc = str(rr[col_desc]).strip() if col_desc else ""
        mrp = str(rr[col_mrp]).strip() if col_mrp else ""
        rate = str(rr[col_rate]).strip() if col_rate else ""
        if not (code or desc):
            continue

        parts = [f"{code} - {desc}"]
        if rate and rate.lower() != "nan":
            parts.append(f"RATE: {rate}")
        if mrp and mrp.lower() != "nan":
            parts.append(f"MRP: {mrp}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)

wa = build_whatsapp_text(out, max_rows=120)
st.text_area("Copy/paste to WhatsApp (first 120 rows)", value=wa, height=180)
st.download_button("Download WhatsApp text (.txt)", data=wa.encode("utf-8"),
                   file_name="raj_catalog_whatsapp.txt", mime="text/plain")