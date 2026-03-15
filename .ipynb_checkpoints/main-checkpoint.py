import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Special Events Enrollment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0f1117;
    color: #e8e6e1;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #181c27 !important;
    border-right: 1px solid #2a2f3e;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #181c27;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #8b8fa8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0ede6 !important;
    font-size: 2rem !important;
    font-family: 'DM Serif Display', serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

/* Headers */
h1 { font-family: 'DM Serif Display', serif !important; color: #f0ede6 !important; }
h2 { font-family: 'DM Serif Display', serif !important; color: #d4c9b8 !important; font-size: 1.4rem !important; }
h3 { font-family: 'DM Serif Display', serif !important; color: #c9bfac !important; font-size: 1.1rem !important; }

/* Section divider label */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a6080;
    font-weight: 600;
    margin-bottom: 4px;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #181c27 !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
[data-baseweb="tab"] {
    color: #8b8fa8 !important;
    font-weight: 500;
}
[aria-selected="true"] {
    background: #2a2f3e !important;
    color: #f0ede6 !important;
    border-radius: 8px;
}

/* Expander */
[data-testid="stExpander"] {
    background: #181c27;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2a2f3e;
    border-radius: 10px;
}

/* Accent pill */
.pill {
    display: inline-block;
    background: #2a2f3e;
    border: 1px solid #3d4460;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    color: #a8b4d8;
    margin: 2px;
}

.highlight-box {
    background: #1a2235;
    border-left: 3px solid #5b8dee;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #c0cce8;
}

.warn-box {
    background: #221a1a;
    border-left: 3px solid #e05b5b;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #e0b4b4;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_COLORS = ["#5b8dee", "#e0905b", "#5bde9a", "#de5b9a", "#dec85b", "#9a5bde", "#5bdecd"]
PLOT_BG = "#0f1117"
PLOT_PAPER = "#0f1117"
PLOT_FONT = "#c0c0d0"
GRID_COLOR = "#1e2235"

def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans, sans-serif", color=PLOT_FONT, size=12),
        legend=dict(bgcolor="#181c27", bordercolor="#2a2f3e", borderwidth=1, font_size=11),
        margin=dict(t=40, b=30, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont_color=PLOT_FONT)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont_color=PLOT_FONT)
    return fig

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", low_memory=False)

    # Normalize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Parse dates
    for col in ["session_starts_at", "enrollment_created_at", "student_created_at",
                "transcript_created_at", "booking_dates"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric coercions
    for col in ["total_consults", "total_booking_amount_90_day", "new_purchase_flag",
                "ly_top_100", "days_to_first_consult", "days_to_first_booking",
                "completed_consult_flag", "attended", "has_exam_transcript",
                "promoting_lead_source_expected_enrollments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean-ish flags
    for col in ["batch_enroll", "enrollee_from_promoting_lead_source_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False, "1": True, "0": False})

    return df

# ── Period helpers ────────────────────────────────────────────────────────────
def assign_period(df):
    """Add a 'period' column: 'FY25' = Oct24–Feb25, 'FY26' = Oct25–Feb26, else NaN."""
    s = df["session_starts_at"]
    fy25 = (s >= "2024-10-01") & (s <= "2025-02-28")
    fy26 = (s >= "2025-10-01") & (s <= "2026-02-28")
    df = df.copy()
    df["period"] = np.where(fy25, "Oct 24 – Feb 25", np.where(fy26, "Oct 25 – Feb 26", None))
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="margin-bottom:2px;">Special Events Enrollment Intelligence</h1>
<p style="color:#5a6080;font-size:0.9rem;margin-top:0;">Year-over-Year Demographic & Performance Analysis · Mock Exams Focus</p>
""", unsafe_allow_html=True)

# Load
try:
    raw = load_data()
except FileNotFoundError:
    st.error("⚠️ `data.csv` not found. Place it in the same directory as this script and rerun.")
    st.stop()

raw = assign_period(raw)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    # Period filter always shown
    period_options = ["Both Periods", "Oct 24 – Feb 25", "Oct 25 – Feb 26"]
    period_filter = st.selectbox("Period", period_options)

    # Event type filter
    if "promotion_type" in raw.columns:
        all_event_types = sorted(raw["promotion_type"].dropna().unique().tolist())
        selected_event_types = st.multiselect(
            "Event / Promotion Type",
            options=all_event_types,
            default=[x for x in all_event_types if "Mock" in str(x)],
        )
    else:
        selected_event_types = []

    # Subject filter
    if "course_subject_name" in raw.columns:
        all_subjects = sorted(raw["course_subject_name"].dropna().unique().tolist())
        selected_subjects = st.multiselect("Subject", options=all_subjects, default=all_subjects)
    else:
        selected_subjects = all_subjects = []

    # Region filter
    if "enrollee_lead_source_region" in raw.columns:
        all_regions = sorted(raw["enrollee_lead_source_region"].dropna().unique().tolist())
        selected_regions = st.multiselect("Region", options=all_regions, default=all_regions)
    else:
        selected_regions = all_regions = []

    # Attendees only?
    attended_only = st.checkbox("Attended Only", value=False)

    st.markdown("---")
    st.markdown("<span class='section-label'>About</span>", unsafe_allow_html=True)
    st.caption("One row per event per enrollment. Events with no enrollments are included. Consults/bookings may be non-unique if assigned to multiple enrollments.")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = raw[raw["period"].notna()].copy()

if selected_event_types:
    df = df[df["promotion_type"].isin(selected_event_types)]
if selected_subjects:
    df = df[df["course_subject_name"].isin(selected_subjects)]
if selected_regions:
    df = df[df["enrollee_lead_source_region"].isin(selected_regions)]
if attended_only and "attended" in df.columns:
    df = df[df["attended"] == 1]

if period_filter != "Both Periods":
    df = df[df["period"] == period_filter]

# Split for YoY
fy25 = df[df["period"] == "Oct 24 – Feb 25"]
fy26 = df[df["period"] == "Oct 25 – Feb 26"]

# ── Top KPIs ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📌 Period Snapshot")

def safe_pct(a, b):
    if b == 0:
        return None
    return round((a - b) / b * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)

n25 = len(fy25)
n26 = len(fy26)
k1.metric("Enrollments · Oct 24–Feb 25", f"{n25:,}")
k2.metric("Enrollments · Oct 25–Feb 26", f"{n26:,}", delta=f"{safe_pct(n26, n25)}%" if n25 else None)

att25 = fy25["attended"].sum() if "attended" in fy25.columns else 0
att26 = fy26["attended"].sum() if "attended" in fy26.columns else 0
k3.metric("Attended · FY25 / FY26", f"{int(att25):,} / {int(att26):,}")

top25 = (fy25["ly_top_100"] == 1).sum() if "ly_top_100" in fy25.columns else 0
top26 = (fy26["ly_top_100"] == 1).sum() if "ly_top_100" in fy26.columns else 0
k4.metric("Top 100 School Enrollments", f"{top25:,} → {top26:,}", delta=f"{safe_pct(top26, top25)}%" if top25 else None)

np25 = fy25["new_purchase_flag"].sum() if "new_purchase_flag" in fy25.columns else 0
np26 = fy26["new_purchase_flag"].sum() if "new_purchase_flag" in fy26.columns else 0
k5.metric("New Purchases (90-day)", f"{int(np25):,} → {int(np26):,}", delta=f"{safe_pct(np26, np25)}%" if np25 else None)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👥 Demographics",
    "📝 Exam Results",
    "🔄 New vs. Returning",
    "🏫 Top 100 Schools",
    "🔍 Purchase Signals",
    "📉 Purchase Driver Analysis",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Demographic Breakdown")

    # Grade distribution
    if "student_grade_at_session_starts_at" in df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Grade Distribution")
            grade_df = (
                df.groupby(["period", "student_grade_at_session_starts_at"])
                .size()
                .reset_index(name="count")
            )
            grade_df["grade"] = pd.to_numeric(grade_df["student_grade_at_session_starts_at"], errors="coerce")
            grade_df = grade_df.dropna(subset=["grade"]).sort_values("grade")
            grade_df["grade_label"] = "Grade " + grade_df["grade"].astype(int).astype(str)

            fig = px.bar(
                grade_df, x="grade_label", y="count", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"count": "Enrollments", "grade_label": ""},
            )
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("### Lead Source Category")
            if "lead_source_category" in df.columns:
                cat_df = (
                    df.groupby(["period", "lead_source_category"])
                    .size()
                    .reset_index(name="count")
                )
                fig2 = px.bar(
                    cat_df, x="lead_source_category", y="count", color="period",
                    barmode="group", color_discrete_sequence=PLOT_COLORS,
                    labels={"count": "Enrollments", "lead_source_category": ""},
                )
                apply_dark_theme(fig2)
                fig2.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig2, use_container_width=True)

    # Enrollee type
    col_c, col_d = st.columns(2)
    with col_c:
        if "enrollee_type" in df.columns:
            st.markdown("### Enrollee Type")
            etype_df = df.groupby(["period", "enrollee_type"]).size().reset_index(name="count")
            fig3 = px.pie(
                etype_df[etype_df["period"] == "Oct 25 – Feb 26"] if period_filter == "Both Periods" else etype_df,
                names="enrollee_type", values="count",
                color_discrete_sequence=PLOT_COLORS,
                title="FY26 Mix" if period_filter == "Both Periods" else "",
            )
            apply_dark_theme(fig3)
            fig3.update_traces(textfont_color="white")
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        if "enrollee_lead_source_region" in df.columns:
            st.markdown("### Region Mix")
            reg_df = (
                df.groupby(["period", "enrollee_lead_source_region"])
                .size()
                .reset_index(name="count")
            )
            fig4 = px.bar(
                reg_df, x="enrollee_lead_source_region", y="count", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"count": "Enrollments", "enrollee_lead_source_region": ""},
            )
            apply_dark_theme(fig4)
            fig4.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig4, use_container_width=True)

    # Subject breakdown
    st.markdown("### Subject Mix YoY")
    if "course_subject_name" in df.columns:
        subj_df = (
            df.groupby(["period", "course_subject_name"])
            .size()
            .reset_index(name="count")
        )
        subj_pct = subj_df.copy()
        total_by_period = subj_pct.groupby("period")["count"].transform("sum")
        subj_pct["pct"] = (subj_pct["count"] / total_by_period * 100).round(1)
        fig5 = px.bar(
            subj_pct, x="pct", y="course_subject_name", color="period",
            barmode="group", orientation="h", color_discrete_sequence=PLOT_COLORS,
            labels={"pct": "% of Period Enrollments", "course_subject_name": ""},
            height=max(400, len(subj_pct["course_subject_name"].unique()) * 28),
        )
        apply_dark_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · EXAM RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Exam & Transcript Outcomes")

    col1, col2, col3 = st.columns(3)
    with col1:
        t25 = fy25["has_exam_transcript"].sum() if "has_exam_transcript" in fy25.columns else 0
        t26 = fy26["has_exam_transcript"].sum() if "has_exam_transcript" in fy26.columns else 0
        st.metric("Has Transcript · FY25", f"{int(t25):,}")
        st.metric("Has Transcript · FY26", f"{int(t26):,}", delta=f"{safe_pct(t26, t25)}%" if t25 else None)
    with col2:
        a25 = fy25["attended"].sum() if "attended" in fy25.columns else 0
        a26 = fy26["attended"].sum() if "attended" in fy26.columns else 0
        att_rate25 = round(a25 / len(fy25) * 100, 1) if len(fy25) else 0
        att_rate26 = round(a26 / len(fy26) * 100, 1) if len(fy26) else 0
        st.metric("Attendance Rate · FY25", f"{att_rate25}%")
        st.metric("Attendance Rate · FY26", f"{att_rate26}%", delta=f"{round(att_rate26 - att_rate25, 1)} pp")
    with col3:
        c25 = fy25["completed_consult_flag"].sum() if "completed_consult_flag" in fy25.columns else 0
        c26 = fy26["completed_consult_flag"].sum() if "completed_consult_flag" in fy26.columns else 0
        st.metric("Completed Consult · FY25", f"{int(c25):,}")
        st.metric("Completed Consult · FY26", f"{int(c26):,}", delta=f"{safe_pct(c26, c25)}%" if c25 else None)

    st.markdown("---")

    # Exam name breakdown
    if "exam_name" in df.columns:
        st.markdown("### Exam Versions Taken")
        exam_df = (
            df[df["exam_name"].notna()]
            .groupby(["period", "exam_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(40)
        )
        fig_ex = px.bar(
            exam_df, y="exam_name", x="count", color="period",
            barmode="group", orientation="h",
            color_discrete_sequence=PLOT_COLORS,
            labels={"count": "# Enrollments with This Exam", "exam_name": ""},
            height=max(400, len(exam_df["exam_name"].unique()) * 26),
        )
        apply_dark_theme(fig_ex)
        st.plotly_chart(fig_ex, use_container_width=True)

    # Attendance by subject
    if "attended" in df.columns and "course_subject_name" in df.columns:
        st.markdown("### Attendance Rate by Subject & Period")
        att_subj = (
            df.groupby(["period", "course_subject_name"])
            .agg(total=("attended", "count"), attended=("attended", "sum"))
            .reset_index()
        )
        att_subj["att_rate"] = (att_subj["attended"] / att_subj["total"] * 100).round(1)
        fig_att = px.bar(
            att_subj, x="att_rate", y="course_subject_name", color="period",
            barmode="group", orientation="h",
            color_discrete_sequence=PLOT_COLORS,
            labels={"att_rate": "Attendance Rate (%)", "course_subject_name": ""},
            height=max(400, len(att_subj["course_subject_name"].unique()) * 28),
        )
        apply_dark_theme(fig_att)
        st.plotly_chart(fig_att, use_container_width=True)

    # Consult outcome breakdown
    if "consult_outcome" in df.columns:
        st.markdown("### Consult Outcomes")
        co_df = (
            df[df["consult_outcome"].notna() & (df["consult_outcome"] != "")]
            .groupby(["period", "consult_outcome"])
            .size()
            .reset_index(name="count")
        )
        if not co_df.empty:
            fig_co = px.bar(
                co_df, x="consult_outcome", y="count", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"count": "# Records", "consult_outcome": ""},
            )
            apply_dark_theme(fig_co)
            st.plotly_chart(fig_co, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · NEW vs. RETURNING
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## New vs. Returning Families")

    st.markdown("""
    <div class="highlight-box">
    <b>Methodology:</b> "New" = <code>enrollment_created_student_lead</code> matches the enrollment ID (student was created at/around enrollment).
    An enrollee is also considered <em>new to mock exams</em> if their first mock enrollment falls within the given period.
    </div>
    """, unsafe_allow_html=True)

    col_nv1, col_nv2 = st.columns(2)

    # Using enrollment_created_student_lead as new indicator
    if "enrollment_created_student_lead" in df.columns:
        df["is_new_student"] = df["enrollment_created_student_lead"].astype(str).str.strip().notna() & \
                               (df["enrollment_created_student_lead"].astype(str).str.strip() != "") & \
                               (df["enrollment_created_student_lead"].astype(str).str.strip() != "nan")
    else:
        df["is_new_student"] = False

    with col_nv1:
        new_df = df.groupby(["period", "is_new_student"]).size().reset_index(name="count")
        new_df["label"] = new_df["is_new_student"].map({True: "New Student Lead", False: "Existing Student"})
        fig_new = px.bar(
            new_df, x="period", y="count", color="label",
            barmode="stack", color_discrete_sequence=PLOT_COLORS,
            labels={"count": "Enrollments", "period": ""},
            title="New vs. Existing Student Leads",
        )
        apply_dark_theme(fig_new)
        st.plotly_chart(fig_new, use_container_width=True)

    with col_nv2:
        # First mock exam enrollment per enrollee
        mock_df = df[df["promotion_type"].astype(str).str.contains("Mock", na=False)].copy() if "promotion_type" in df.columns else df.copy()
        if "enrollee_id" in mock_df.columns and "session_starts_at" in mock_df.columns:
            first_mock = (
                mock_df.sort_values("session_starts_at")
                .groupby("enrollee_id")["session_starts_at"]
                .min()
                .reset_index()
                .rename(columns={"session_starts_at": "first_mock_date"})
            )
            mock_df = mock_df.merge(first_mock, on="enrollee_id", how="left")
            mock_df["first_time_mock"] = mock_df["session_starts_at"] == mock_df["first_mock_date"]

            first_time_df = mock_df.groupby(["period", "first_time_mock"]).size().reset_index(name="count")
            first_time_df["label"] = first_time_df["first_time_mock"].map({True: "First Mock Enrollment", False: "Returning Mock Enrollee"})
            fig_ft = px.bar(
                first_time_df, x="period", y="count", color="label",
                barmode="stack", color_discrete_sequence=[PLOT_COLORS[2], PLOT_COLORS[1]],
                labels={"count": "Enrollments", "period": ""},
                title="First-Time vs. Returning Mock Enrollees",
            )
            apply_dark_theme(fig_ft)
            st.plotly_chart(fig_ft, use_container_width=True)

    # Table: summary
    st.markdown("### Period Summary Table")
    rows = []
    for p, grp in df.groupby("period"):
        n = len(grp)
        n_new = grp["is_new_student"].sum()
        n_att = grp["attended"].sum() if "attended" in grp.columns else 0
        n_consult = grp["completed_consult_flag"].sum() if "completed_consult_flag" in grp.columns else 0
        n_top100 = (grp["ly_top_100"] == 1).sum() if "ly_top_100" in grp.columns else 0
        rows.append({
            "Period": p,
            "Total Enrollments": n,
            "New Student Leads": int(n_new),
            "% New": f"{round(n_new/n*100,1)}%" if n else "—",
            "Attended": int(n_att),
            "Attendance Rate": f"{round(n_att/n*100,1)}%" if n else "—",
            "Completed Consult": int(n_consult),
            "Top 100 School": int(n_top100),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Period"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · TOP 100 SCHOOLS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Top 100 School Analysis")

    if "ly_top_100" not in df.columns:
        st.warning("Column `ly_top_100` not found in data.")
    else:
        df["top100_label"] = df["ly_top_100"].map({1: "Top 100 School", 0: "Non-Top-100"}).fillna("Unknown")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            t100_period = (
                df.groupby(["period", "top100_label"])
                .size()
                .reset_index(name="count")
            )
            fig_t100 = px.bar(
                t100_period, x="period", y="count", color="top100_label",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"count": "Enrollments", "period": ""},
                title="Top 100 vs. Non-Top-100 by Period",
            )
            apply_dark_theme(fig_t100)
            st.plotly_chart(fig_t100, use_container_width=True)

        with col_t2:
            # Top 100 attendance rate
            att_t100 = (
                df.groupby(["period", "top100_label"])
                .agg(total=("attended", "count"), attended=("attended", "sum"))
                .reset_index()
            )
            att_t100["att_rate"] = (att_t100["attended"] / att_t100["total"] * 100).round(1)
            fig_att_t100 = px.bar(
                att_t100, x="top100_label", y="att_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"att_rate": "Attendance Rate (%)", "top100_label": ""},
                title="Attendance Rate by School Tier",
            )
            apply_dark_theme(fig_att_t100)
            st.plotly_chart(fig_att_t100, use_container_width=True)

        # Top schools table
        st.markdown("### Top Enrolling Schools by Period")
        if "enrollee_lead_source_name" in df.columns:
            top_schools = (
                df[df["ly_top_100"] == 1]
                .groupby(["period", "enrollee_lead_source_name"])
                .agg(
                    enrollments=("enrollment_id", "count"),
                    attended=("attended", "sum"),
                    new_purchases=("new_purchase_flag", "sum"),
                )
                .reset_index()
                .sort_values(["period", "enrollments"], ascending=[True, False])
            )
            for p in top_schools["period"].unique():
                with st.expander(f"📍 {p} — Top 100 Schools"):
                    st.dataframe(
                        top_schools[top_schools["period"] == p]
                        .drop("period", axis=1)
                        .head(25)
                        .set_index("enrollee_lead_source_name"),
                        use_container_width=True,
                    )

        # Top 100 vs purchase rate
        st.markdown("### Top 100 School Families: Purchase Rate YoY")
        pur_t100 = (
            df.groupby(["period", "top100_label"])
            .agg(
                enrollments=("enrollment_id", "count"),
                new_purchases=("new_purchase_flag", "sum"),
                booking_amount=("total_booking_amount_90_day", "sum"),
            )
            .reset_index()
        )
        pur_t100["purchase_rate"] = (pur_t100["new_purchases"] / pur_t100["enrollments"] * 100).round(1)
        fig_pur = px.bar(
            pur_t100, x="top100_label", y="purchase_rate", color="period",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={"purchase_rate": "New Purchase Rate (%)", "top100_label": ""},
            title="New Purchase Rate by School Tier & Period",
        )
        apply_dark_theme(fig_pur)
        st.plotly_chart(fig_pur, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · PURCHASE SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## Purchase Signals & Conversion Insights")

    st.markdown("""
    <div class="highlight-box">
    Investigating factors that could explain differences in purchase behavior between the two periods.
    Consult counts may be inflated if one consult maps to multiple enrollments.
    </div>
    """, unsafe_allow_html=True)

    # Booking amount
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if "total_booking_amount_90_day" in df.columns:
            bk_df = (
                df.groupby("period")["total_booking_amount_90_day"]
                .agg(["sum", "mean", "count"])
                .reset_index()
            )
            bk_df.columns = ["Period", "Total Booking $", "Avg Booking $", "Records"]
            st.markdown("### 90-Day Booking Amount")
            fig_bk = px.bar(
                bk_df, x="Period", y="Total Booking $",
                color="Period", color_discrete_sequence=PLOT_COLORS,
                text_auto=".2s",
            )
            apply_dark_theme(fig_bk)
            st.plotly_chart(fig_bk, use_container_width=True)

    with col_p2:
        if "days_to_first_booking" in df.columns:
            st.markdown("### Days to First Booking Distribution")
            dtb = df[df["days_to_first_booking"].notna() & (df["days_to_first_booking"] > 0)]
            fig_dtb = px.box(
                dtb, x="period", y="days_to_first_booking",
                color="period", color_discrete_sequence=PLOT_COLORS,
                labels={"days_to_first_booking": "Days to First Booking", "period": ""},
            )
            apply_dark_theme(fig_dtb)
            st.plotly_chart(fig_dtb, use_container_width=True)

    # Consult analysis
    st.markdown("### Consult Activity")
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        if "total_consults" in df.columns:
            consult_dist = (
                df.groupby("period")["total_consults"]
                .value_counts()
                .reset_index(name="count")
            )
            consult_dist = consult_dist[consult_dist["total_consults"] <= 5]
            fig_cd = px.bar(
                consult_dist, x="total_consults", y="count", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"total_consults": "# Consults", "count": "Enrollments"},
                title="Consult Count Distribution (0–5)",
            )
            apply_dark_theme(fig_cd)
            st.plotly_chart(fig_cd, use_container_width=True)

    with col_p4:
        if "days_to_first_consult" in df.columns:
            dtc = df[df["days_to_first_consult"].notna() & (df["days_to_first_consult"] >= 0)]
            fig_dtc = px.histogram(
                dtc, x="days_to_first_consult", color="period",
                nbins=40, barmode="overlay",
                color_discrete_sequence=PLOT_COLORS,
                opacity=0.75,
                labels={"days_to_first_consult": "Days to First Consult"},
                title="Days to First Consult (Histogram)",
            )
            apply_dark_theme(fig_dtc)
            st.plotly_chart(fig_dtc, use_container_width=True)

    # Promotion type mix
    if "promotion_type" in df.columns:
        st.markdown("### Promotion Type Mix YoY")
        promo_df = df.groupby(["period", "promotion_type"]).size().reset_index(name="count")
        total_p = promo_df.groupby("period")["count"].transform("sum")
        promo_df["pct"] = (promo_df["count"] / total_p * 100).round(1)
        fig_promo = px.bar(
            promo_df, x="promotion_type", y="pct", color="period",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={"pct": "% of Period Enrollments", "promotion_type": ""},
        )
        apply_dark_theme(fig_promo)
        fig_promo.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_promo, use_container_width=True)

    # Advisor / PM breakdown — who's driving enrollments
    if "enrollee_lead_source_advisor_name" in df.columns:
        st.markdown("### Top Advisors by Enrollment Volume")
        adv_df = (
            df.groupby(["period", "enrollee_lead_source_advisor_name"])
            .agg(
                enrollments=("enrollment_id", "count"),
                attended=("attended", "sum"),
                new_purchases=("new_purchase_flag", "sum"),
            )
            .reset_index()
            .sort_values(["period", "enrollments"], ascending=[True, False])
        )
        for p in adv_df["period"].unique():
            with st.expander(f"📋 {p} — Top Advisors"):
                st.dataframe(
                    adv_df[adv_df["period"] == p].drop("period", axis=1).head(20).set_index("enrollee_lead_source_advisor_name"),
                    use_container_width=True,
                )

    # Key divergences callout (kept as quick summary)
    st.markdown("---")
    st.markdown("""
    <div class="highlight-box">
    💡 See the <b>📉 Purchase Driver Analysis</b> tab for a full visual breakdown of each potential driver.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · PURCHASE DRIVER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("## 📉 Purchase Driver Analysis")
    st.markdown("""
    <div class="warn-box">
    Each section below tests one potential driver of lower purchases. Charts compare <b>Oct 24–Feb 25</b> vs. <b>Oct 25–Feb 26</b>.
    A divergence in any dimension is a signal worth investigating further.
    </div>
    """, unsafe_allow_html=True)

    # ── DRIVER 1: Grade Mix Shift ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1 · Grade Mix — Are We Skewing Younger?")
    st.caption("Younger students (grades 9–10) have lower urgency to purchase immediately. A shift toward younger grades in FY26 could suppress conversion.")

    if "student_grade_at_session_starts_at" in df.columns:
        grade_raw = df.copy()
        grade_raw["grade"] = pd.to_numeric(grade_raw["student_grade_at_session_starts_at"], errors="coerce")
        grade_raw = grade_raw.dropna(subset=["grade"])
        grade_raw["grade_label"] = "Gr " + grade_raw["grade"].astype(int).astype(str)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # % of period enrollments by grade
            gd = grade_raw.groupby(["period", "grade_label"]).size().reset_index(name="count")
            gd["pct"] = gd["count"] / gd.groupby("period")["count"].transform("sum") * 100
            fig_g = px.bar(
                gd, x="grade_label", y="pct", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"pct": "% of Period Enrollments", "grade_label": "Grade"},
                title="Grade Mix % by Period",
            )
            apply_dark_theme(fig_g)
            st.plotly_chart(fig_g, use_container_width=True)

        with col_g2:
            # New purchase rate by grade + period
            pur_grade = (
                grade_raw.groupby(["period", "grade_label"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            pur_grade["purchase_rate"] = (pur_grade["purchases"] / pur_grade["enrollments"] * 100).round(1)
            fig_pg = px.bar(
                pur_grade, x="grade_label", y="purchase_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"purchase_rate": "New Purchase Rate (%)", "grade_label": "Grade"},
                title="Purchase Rate by Grade & Period",
            )
            apply_dark_theme(fig_pg)
            st.plotly_chart(fig_pg, use_container_width=True)

        # Avg grade per period scorecard
        avg_grades = grade_raw.groupby("period")["grade"].mean().reset_index()
        cols_g = st.columns(len(avg_grades))
        for i, row in avg_grades.iterrows():
            cols_g[i].metric(f"Avg Grade · {row['period']}", f"{row['grade']:.2f}")

    # ── DRIVER 2: Top 100 School Mix ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2 · Top 100 School Mix — Fewer High-LTV Families?")
    st.caption("Top 100 school families historically have higher booking amounts. A drop in their share = lower-value pipeline mix.")

    if "ly_top_100" in df.columns:
        col_t1, col_t2, col_t3 = st.columns(3)

        top100_period = (
            df.groupby("period")
            .agg(total=("enrollment_id", "count"), top100=("ly_top_100", "sum"))
            .reset_index()
        )
        top100_period["top100_pct"] = (top100_period["top100"] / top100_period["total"] * 100).round(1)

        for i, row in top100_period.iterrows():
            col = [col_t1, col_t2][i % 2]
            col.metric(
                f"Top 100 Share · {row['period']}",
                f"{row['top100_pct']}%",
                delta=None if i == 0 else f"{round(row['top100_pct'] - top100_period.iloc[0]['top100_pct'], 1)} pp"
            )

        # Avg 90-day booking by tier
        if "total_booking_amount_90_day" in df.columns:
            with col_t3:
                bk_tier = (
                    df[df["ly_top_100"].isin([0, 1])]
                    .groupby(["period", "ly_top_100"])["total_booking_amount_90_day"]
                    .mean()
                    .reset_index()
                )
                bk_tier["tier"] = bk_tier["ly_top_100"].map({1: "Top 100", 0: "Non-Top-100"})
                st.dataframe(
                    bk_tier[["period", "tier", "total_booking_amount_90_day"]]
                    .rename(columns={"total_booking_amount_90_day": "Avg 90-day Booking $"})
                    .set_index("period"),
                    use_container_width=True,
                )

        fig_t100_share = px.bar(
            top100_period, x="period", y="top100_pct",
            color="period", color_discrete_sequence=PLOT_COLORS,
            text="top100_pct",
            labels={"top100_pct": "% Top 100 School Enrollments", "period": ""},
            title="Top 100 School Share of Enrollments",
        )
        apply_dark_theme(fig_t100_share)
        fig_t100_share.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_t100_share, use_container_width=True)

    # ── DRIVER 3: Attendance Rate ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3 · Attendance Rate — Less Product Exposure?")
    st.caption("Students who don't show up never experience the product. Lower attendance → fewer organic upsell moments.")

    if "attended" in df.columns:
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            att_overall = (
                df.groupby("period")
                .agg(total=("attended", "count"), attended=("attended", "sum"))
                .reset_index()
            )
            att_overall["att_rate"] = (att_overall["attended"] / att_overall["total"] * 100).round(1)
            fig_att_ov = px.bar(
                att_overall, x="period", y="att_rate",
                color="period", color_discrete_sequence=PLOT_COLORS,
                text="att_rate",
                labels={"att_rate": "Attendance Rate (%)", "period": ""},
                title="Overall Attendance Rate by Period",
            )
            apply_dark_theme(fig_att_ov)
            fig_att_ov.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_att_ov, use_container_width=True)

        with col_a2:
            # Purchase rate: attended vs. not attended
            att_pur = (
                df.groupby(["period", "attended"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            att_pur = att_pur[att_pur["attended"].isin([0, 1])]
            att_pur["purchase_rate"] = (att_pur["purchases"] / att_pur["enrollments"] * 100).round(1)
            att_pur["attended_label"] = att_pur["attended"].map({1: "Attended", 0: "Did Not Attend"})
            fig_att_pur = px.bar(
                att_pur, x="attended_label", y="purchase_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"purchase_rate": "Purchase Rate (%)", "attended_label": ""},
                title="Purchase Rate: Attended vs. Not Attended",
            )
            apply_dark_theme(fig_att_pur)
            st.plotly_chart(fig_att_pur, use_container_width=True)

    # ── DRIVER 4: Completed Consults ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4 · Completed Consults — Fewer Advisor Touchpoints?")
    st.caption("Consults are the primary sales motion. Fewer completed consults = less pipeline activation.")

    if "completed_consult_flag" in df.columns:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            consult_rate = (
                df.groupby("period")
                .agg(total=("enrollment_id", "count"), consults=("completed_consult_flag", "sum"))
                .reset_index()
            )
            consult_rate["consult_rate"] = (consult_rate["consults"] / consult_rate["total"] * 100).round(1)
            fig_cr = px.bar(
                consult_rate, x="period", y="consult_rate",
                color="period", color_discrete_sequence=PLOT_COLORS,
                text="consult_rate",
                labels={"consult_rate": "Completed Consult Rate (%)", "period": ""},
                title="Consult Completion Rate by Period",
            )
            apply_dark_theme(fig_cr)
            fig_cr.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_cr, use_container_width=True)

        with col_c2:
            # Purchase rate with vs without consult
            consult_pur = (
                df.groupby(["period", "completed_consult_flag"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            consult_pur = consult_pur[consult_pur["completed_consult_flag"].isin([0, 1])]
            consult_pur["purchase_rate"] = (consult_pur["purchases"] / consult_pur["enrollments"] * 100).round(1)
            consult_pur["consult_label"] = consult_pur["completed_consult_flag"].map({1: "Had Consult", 0: "No Consult"})
            fig_cp = px.bar(
                consult_pur, x="consult_label", y="purchase_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"purchase_rate": "Purchase Rate (%)", "consult_label": ""},
                title="Purchase Rate: With vs. Without Consult",
            )
            apply_dark_theme(fig_cp)
            st.plotly_chart(fig_cp, use_container_width=True)

    # ── DRIVER 5: Days to First Consult ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5 · Days to First Consult — Are Leads Going Cold?")
    st.caption("The longer it takes to reach a lead after a mock exam, the colder they become. An increase in median days = lost urgency.")

    if "days_to_first_consult" in df.columns:
        col_d1, col_d2 = st.columns(2)
        dtc_data = df[df["days_to_first_consult"].notna() & (df["days_to_first_consult"] >= 0)]

        with col_d1:
            # Median days to consult by period
            dtc_summary = (
                dtc_data.groupby("period")["days_to_first_consult"]
                .agg(["median", "mean", "count"])
                .reset_index()
            )
            dtc_summary.columns = ["Period", "Median Days", "Mean Days", "N"]
            for i, row in dtc_summary.iterrows():
                st.metric(
                    f"Median Days to Consult · {row['Period']}",
                    f"{row['Median Days']:.1f} days",
                    delta=None if i == 0 else f"{round(row['Median Days'] - dtc_summary.iloc[0]['Median Days'], 1)} days"
                )

        with col_d2:
            # Box plot distribution
            fig_dtc2 = px.box(
                dtc_data, x="period", y="days_to_first_consult",
                color="period", color_discrete_sequence=PLOT_COLORS,
                labels={"days_to_first_consult": "Days to First Consult", "period": ""},
                title="Days to First Consult Distribution",
            )
            apply_dark_theme(fig_dtc2)
            st.plotly_chart(fig_dtc2, use_container_width=True)

        # Bucket analysis: 0-7 / 8-30 / 31-60 / 60+
        dtc_data = dtc_data.copy()
        dtc_data["consult_speed"] = pd.cut(
            dtc_data["days_to_first_consult"],
            bins=[-1, 7, 30, 60, 9999],
            labels=["0–7 days", "8–30 days", "31–60 days", "60+ days"]
        )
        speed_df = dtc_data.groupby(["period", "consult_speed"]).size().reset_index(name="count")
        speed_pct = speed_df.copy()
        speed_pct["pct"] = speed_pct["count"] / speed_pct.groupby("period")["count"].transform("sum") * 100
        fig_speed = px.bar(
            speed_pct, x="consult_speed", y="pct", color="period",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={"pct": "% of Consults", "consult_speed": "Speed Bucket"},
            title="Consult Speed Buckets by Period (% share)",
        )
        apply_dark_theme(fig_speed)
        st.plotly_chart(fig_speed, use_container_width=True)

    # ── DRIVER 6: Promotion Type Mix ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 6 · Promotion Type Mix — More Unpromoted Events?")
    st.caption("Unpromoted events attract self-selected students with no advisor touchpoint. A shift toward unpromoted = less warm pipeline.")

    if "promotion_type" in df.columns:
        col_pr1, col_pr2 = st.columns(2)
        with col_pr1:
            promo_mix = df.groupby(["period", "promotion_type"]).size().reset_index(name="count")
            promo_mix["pct"] = promo_mix["count"] / promo_mix.groupby("period")["count"].transform("sum") * 100
            fig_pm = px.bar(
                promo_mix, x="period", y="pct", color="promotion_type",
                barmode="stack", color_discrete_sequence=PLOT_COLORS,
                labels={"pct": "% of Enrollments", "promotion_type": "Type"},
                title="Promotion Type Stack (% share)",
            )
            apply_dark_theme(fig_pm)
            st.plotly_chart(fig_pm, use_container_width=True)

        with col_pr2:
            # Purchase rate by promotion type
            promo_pur = (
                df.groupby(["period", "promotion_type"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            promo_pur["purchase_rate"] = (promo_pur["purchases"] / promo_pur["enrollments"] * 100).round(1)
            fig_pp = px.bar(
                promo_pur, x="promotion_type", y="purchase_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"purchase_rate": "Purchase Rate (%)", "promotion_type": ""},
                title="Purchase Rate by Promotion Type",
            )
            apply_dark_theme(fig_pp)
            fig_pp.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_pp, use_container_width=True)

    # ── DRIVER 7: Regional Composition ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 7 · Regional Composition — Over-Indexing on Low-Converting Regions?")
    st.caption("If regions with historically lower purchase rates grew their share in FY26, the aggregate conversion rate drops even if per-region rates held steady.")

    if "enrollee_lead_source_region" in df.columns:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            reg_mix = df.groupby(["period", "enrollee_lead_source_region"]).size().reset_index(name="count")
            reg_mix["pct"] = reg_mix["count"] / reg_mix.groupby("period")["count"].transform("sum") * 100
            fig_rm = px.bar(
                reg_mix, x="enrollee_lead_source_region", y="pct", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"pct": "% of Period Enrollments", "enrollee_lead_source_region": ""},
                title="Regional Share by Period (%)",
            )
            apply_dark_theme(fig_rm)
            fig_rm.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_rm, use_container_width=True)

        with col_r2:
            # Purchase rate by region + period
            reg_pur = (
                df.groupby(["period", "enrollee_lead_source_region"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            reg_pur["purchase_rate"] = (reg_pur["purchases"] / reg_pur["enrollments"] * 100).round(1)
            fig_rp = px.bar(
                reg_pur, x="enrollee_lead_source_region", y="purchase_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"purchase_rate": "Purchase Rate (%)", "enrollee_lead_source_region": ""},
                title="Purchase Rate by Region & Period",
            )
            apply_dark_theme(fig_rp)
            fig_rp.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_rp, use_container_width=True)

        # Bubble chart: share shift vs. purchase rate
        st.markdown("#### Region Shift Map: Volume Change vs. Purchase Rate")
        st.caption("Bubble size = FY26 enrollment count. X = change in regional share (pp). Y = FY26 purchase rate. Regions in the bottom-right are growing AND underperforming.")
        reg_pivot = reg_pur.pivot(index="enrollee_lead_source_region", columns="period", values="purchase_rate").reset_index()
        reg_share = reg_mix.pivot(index="enrollee_lead_source_region", columns="period", values="pct").reset_index()
        reg_count = reg_mix.pivot(index="enrollee_lead_source_region", columns="period", values="count").reset_index()

        p25 = "Oct 24 – Feb 25"
        p26 = "Oct 25 – Feb 26"

        if p25 in reg_pivot.columns and p26 in reg_pivot.columns:
            bubble = reg_pivot.rename(columns={p25: "pur_fy25", p26: "pur_fy26"})
            if p25 in reg_share.columns and p26 in reg_share.columns:
                share_diff = reg_share[p26].fillna(0) - reg_share[p25].fillna(0)
                bubble["share_shift_pp"] = share_diff.values
            if p26 in reg_count.columns:
                bubble["fy26_count"] = reg_count[p26].fillna(0).values

            bubble = bubble.dropna(subset=["pur_fy26"])
            if not bubble.empty:
                fig_bub = px.scatter(
                    bubble,
                    x="share_shift_pp",
                    y="pur_fy26",
                    size="fy26_count",
                    text="enrollee_lead_source_region",
                    color_discrete_sequence=PLOT_COLORS,
                    labels={
                        "share_shift_pp": "Change in Regional Share (pp, FY25→FY26)",
                        "pur_fy26": "FY26 Purchase Rate (%)",
                    },
                    title="Region Shift vs. Purchase Rate (FY26)",
                    size_max=50,
                )
                apply_dark_theme(fig_bub)
                fig_bub.update_traces(textposition="top center", textfont_size=10)
                fig_bub.add_vline(x=0, line_dash="dash", line_color="#5a6080")
                fig_bub.add_hline(
                    y=reg_pur[reg_pur["period"] == p26]["purchase_rate"].mean(),
                    line_dash="dash", line_color="#5a6080",
                    annotation_text="Avg purchase rate",
                    annotation_font_color="#8b8fa8",
                )
                st.plotly_chart(fig_bub, use_container_width=True)

    # ── Summary Scorecard ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧾 Driver Summary Scorecard")
    st.caption("Quick YoY comparison across all 7 dimensions. Red = potential concern.")

    summary_rows = []

    def period_val(grp_df, col, agg="mean"):
        try:
            v = grp_df[col].agg(agg)
            return round(float(v), 2) if not pd.isna(v) else None
        except Exception:
            return None

    for p, grp in [(p25, fy25), (p26, fy26)]:
        grade_avg = pd.to_numeric(grp.get("student_grade_at_session_starts_at", pd.Series()), errors="coerce").mean()
        top100_sh = (grp["ly_top_100"] == 1).sum() / len(grp) * 100 if len(grp) else None
        att_rate  = grp["attended"].sum() / len(grp) * 100 if "attended" in grp.columns and len(grp) else None
        consult_r = grp["completed_consult_flag"].sum() / len(grp) * 100 if "completed_consult_flag" in grp.columns and len(grp) else None
        dtc_med   = grp["days_to_first_consult"].median() if "days_to_first_consult" in grp.columns else None
        unprom_sh = (grp["promotion_type"].astype(str).str.contains("Unpromoted", na=False)).sum() / len(grp) * 100 if "promotion_type" in grp.columns and len(grp) else None
        pur_rate  = grp["new_purchase_flag"].sum() / len(grp) * 100 if "new_purchase_flag" in grp.columns and len(grp) else None

        summary_rows.append({
            "Period": p,
            "Avg Grade": round(grade_avg, 2) if not pd.isna(grade_avg) else "—",
            "Top 100 Share %": f"{round(top100_sh,1)}%" if top100_sh is not None else "—",
            "Attendance Rate %": f"{round(att_rate,1)}%" if att_rate is not None else "—",
            "Consult Rate %": f"{round(consult_r,1)}%" if consult_r is not None else "—",
            "Median Days to Consult": f"{round(dtc_med,1)}" if dtc_med is not None and not pd.isna(dtc_med) else "—",
            "Unpromoted Share %": f"{round(unprom_sh,1)}%" if unprom_sh is not None else "—",
            "New Purchase Rate %": f"{round(pur_rate,1)}%" if pur_rate is not None else "—",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("Period"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Special Events Enrollment Intelligence Dashboard · Built with Streamlit & Plotly · Data: data.csv")