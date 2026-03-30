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

    # ── Month filter ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<span class='section-label'>Month Range (applied to both periods)</span>", unsafe_allow_html=True)
    st.caption("Select which calendar months to include. Use this to compare apples-to-apples — e.g. Oct only, or Oct–Jan.")

    # The two periods span Oct–Feb, so those are the only relevant months
    PERIOD_MONTHS = {
        10: "October",
        11: "November",
        12: "December",
        1:  "January",
        2:  "February",
    }
    MONTH_ORDER = [10, 11, 12, 1, 2]  # chronological within the Oct–Feb window

    selected_months = st.multiselect(
        "Include Months",
        options=MONTH_ORDER,
        default=MONTH_ORDER,
        format_func=lambda m: PERIOD_MONTHS[m],
    )
    if not selected_months:
        selected_months = MONTH_ORDER  # safety: never filter to nothing

    # Show a small human-readable summary of what's selected
    month_labels = [PERIOD_MONTHS[m] for m in MONTH_ORDER if m in selected_months]
    st.caption(f"✅ Included: {', '.join(month_labels)}")

    st.markdown("---")

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

# Month filter — applied symmetrically to both periods
df = df[df["session_starts_at"].dt.month.isin(selected_months)]

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

# Show active month range banner just below header
_active_months_str = ", ".join([PERIOD_MONTHS[m] for m in MONTH_ORDER if m in selected_months])
if selected_months != MONTH_ORDER:
    st.info(f"📅 **Month filter active:** showing only **{_active_months_str}** from each period. Both periods filtered identically.")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "👥 Demographics",
    "📝 Exam Results",
    "🔄 New vs. Returning",
    "🏫 Top 100 Schools",
    "🔍 Purchase Signals",
    "📉 Purchase Driver Analysis",
    "🔬 Deep Dive",
    "⏱️ Timing & Cadence",
    "🗺️ SAT Aug–Dec Regional",
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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 · DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("## 🔬 Deep Dive Analysis")
    st.markdown("""
    <div class="highlight-box">
    Three targeted investigations: close rate by exam type &amp; region, consult-but-no-purchase population analysis,
    and the Atlantic / Midwest / South regional sales drop.
    </div>
    """, unsafe_allow_html=True)

    dd_p25 = "Oct 24 – Feb 25"
    dd_p26 = "Oct 25 – Feb 26"

    # ── Helper: classify exam bucket ─────────────────────────────────────────
    def exam_bucket(subject):
        s = str(subject).upper()
        if "ACT" in s:
            return "ACT"
        elif "PSAT" in s or "NMSQT" in s:
            return "PSAT/NMSQT"
        elif "SAT" in s:
            return "SAT"
        elif "AP " in s or s.startswith("AP"):
            return "AP"
        else:
            return "Other"

    df["exam_bucket"] = df["course_subject_name"].apply(exam_bucket) if "course_subject_name" in df.columns else "Unknown"

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 · CLOSE RATE BY EXAM TYPE & REGION
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 1 · Close Rate by Exam Type & Region")
    st.caption("'Close rate' = % of enrollments that resulted in a new purchase within 90 days. Filtered to ACT, SAT, PSAT rows only unless you've changed the sidebar subject filter.")

    act_sat_psat = df[df["exam_bucket"].isin(["ACT", "SAT", "PSAT/NMSQT"])].copy()

    if act_sat_psat.empty:
        st.warning("No ACT / SAT / PSAT rows found after current filters.")
    else:
        # ── 1a: Close rate by exam type YoY
        st.markdown("#### Close Rate by Exam Type")
        cr_exam = (
            act_sat_psat.groupby(["period", "exam_bucket"])
            .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
            .reset_index()
        )
        cr_exam["close_rate"] = (cr_exam["purchases"] / cr_exam["enrollments"] * 100).round(1)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_cr_exam = px.bar(
                cr_exam, x="exam_bucket", y="close_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                text="close_rate",
                labels={"close_rate": "Close Rate (%)", "exam_bucket": "Exam Type"},
                title="Close Rate by Exam Type & Period",
            )
            apply_dark_theme(fig_cr_exam)
            fig_cr_exam.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_cr_exam, use_container_width=True)

        with col_e2:
            # Volume context — enrollments per exam type
            fig_vol_exam = px.bar(
                cr_exam, x="exam_bucket", y="enrollments", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"enrollments": "# Enrollments", "exam_bucket": "Exam Type"},
                title="Enrollment Volume by Exam Type & Period",
            )
            apply_dark_theme(fig_vol_exam)
            st.plotly_chart(fig_vol_exam, use_container_width=True)

        # ── 1b: Close rate by exam type × region heatmap
        st.markdown("#### Close Rate by Exam Type × Region (Heatmap)")
        st.caption("Read across a row to see which exam type closes best in each region. Read down a column to compare regions for a given exam.")

        dd_period_sel = st.radio(
            "Period to display in heatmap",
            ["Both (combined)", dd_p25, dd_p26],
            horizontal=True,
            key="heatmap_period",
        )

        hmap_df = act_sat_psat.copy()
        if dd_period_sel != "Both (combined)":
            hmap_df = hmap_df[hmap_df["period"] == dd_period_sel]

        if "enrollee_lead_source_region" in hmap_df.columns:
            hmap_agg = (
                hmap_df.groupby(["enrollee_lead_source_region", "exam_bucket"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            hmap_agg["close_rate"] = (hmap_agg["purchases"] / hmap_agg["enrollments"] * 100).round(1)
            hmap_pivot = hmap_agg.pivot(index="enrollee_lead_source_region", columns="exam_bucket", values="close_rate").fillna(0)

            fig_hmap = px.imshow(
                hmap_pivot,
                text_auto=".1f",
                color_continuous_scale="Blues",
                labels={"color": "Close Rate (%)"},
                title=f"Close Rate Heatmap — {dd_period_sel}",
                aspect="auto",
            )
            fig_hmap.update_layout(
                paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
                font=dict(family="DM Sans, sans-serif", color=PLOT_FONT, size=11),
                margin=dict(t=50, b=30, l=10, r=10),
                coloraxis_colorbar=dict(tickfont_color=PLOT_FONT, title_font_color=PLOT_FONT),
            )
            st.plotly_chart(fig_hmap, use_container_width=True)

        # ── 1c: YoY close rate delta by region × exam
        st.markdown("#### YoY Close Rate Change by Region × Exam Type")
        st.caption("Negative = worse in FY26. Positive = improved. Gray = insufficient data in one period.")

        if "enrollee_lead_source_region" in act_sat_psat.columns:
            delta_agg = (
                act_sat_psat.groupby(["period", "enrollee_lead_source_region", "exam_bucket"])
                .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
                .reset_index()
            )
            delta_agg["close_rate"] = (delta_agg["purchases"] / delta_agg["enrollments"] * 100).round(1)

            fy25_cr = delta_agg[delta_agg["period"] == dd_p25].set_index(["enrollee_lead_source_region", "exam_bucket"])["close_rate"]
            fy26_cr = delta_agg[delta_agg["period"] == dd_p26].set_index(["enrollee_lead_source_region", "exam_bucket"])["close_rate"]
            delta_series = (fy26_cr - fy25_cr).reset_index()
            delta_series.columns = ["enrollee_lead_source_region", "exam_bucket", "close_rate_delta"]
            delta_pivot = delta_series.pivot(index="enrollee_lead_source_region", columns="exam_bucket", values="close_rate_delta")

            fig_delta = px.imshow(
                delta_pivot,
                text_auto=".1f",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"color": "Δ Close Rate (pp)"},
                title="YoY Close Rate Change (FY26 minus FY25, pp)",
                aspect="auto",
            )
            fig_delta.update_layout(
                paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
                font=dict(family="DM Sans, sans-serif", color=PLOT_FONT, size=11),
                margin=dict(t=50, b=30, l=10, r=10),
                coloraxis_colorbar=dict(tickfont_color=PLOT_FONT, title_font_color=PLOT_FONT),
            )
            st.plotly_chart(fig_delta, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 · CONSULT — PURCHASED vs. DID NOT PURCHASE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 2 · Had a Consult: Purchased vs. Did Not Purchase")
    st.caption("Two lenses: (A) what separates converters from non-converters overall, and (B) how the non-converting consult population has shifted from FY25 → FY26.")

    consult_pop = df[df["completed_consult_flag"] == 1].copy() if "completed_consult_flag" in df.columns else pd.DataFrame()

    if consult_pop.empty:
        st.warning("No completed consult records found after current filters.")
    else:
        consult_pop["purchased"] = consult_pop["new_purchase_flag"].fillna(0).astype(int)
        consult_pop["purchase_label"] = consult_pop["purchased"].map({1: "✅ Purchased", 0: "❌ Did Not Purchase"})

        # ── Top-line KPIs split by period ─────────────────────────────────────
        st.markdown("#### Top-Line: Consult → Purchase Rate by Period")
        kpi_cols = st.columns(6)
        for i, (p, grp) in enumerate([(dd_p25, consult_pop[consult_pop["period"] == dd_p25]),
                                       (dd_p26, consult_pop[consult_pop["period"] == dd_p26])]):
            n_total = len(grp)
            n_pur   = (grp["purchased"] == 1).sum()
            n_nopur = (grp["purchased"] == 0).sum()
            rate    = round(n_pur / n_total * 100, 1) if n_total else 0
            offset  = i * 3
            kpi_cols[offset].metric(f"Consult Population · {p[:6]}", f"{n_total:,}")
            kpi_cols[offset+1].metric(f"Purchased · {p[:6]}", f"{n_pur:,}")
            kpi_cols[offset+2].metric(
                f"Conversion Rate · {p[:6]}",
                f"{rate}%",
                delta=None if i == 0 else f"{round(rate - (consult_pop[consult_pop['period']==dd_p25]['purchased'].sum() / len(consult_pop[consult_pop['period']==dd_p25]) * 100 if len(consult_pop[consult_pop['period']==dd_p25]) else 0), 1)} pp"
            )

        # ── Consult conversion rate trend bar
        conv_trend = (
            consult_pop.groupby("period")
            .agg(total=("purchased", "count"), purchased=("purchased", "sum"))
            .reset_index()
        )
        conv_trend["conv_rate"] = (conv_trend["purchased"] / conv_trend["total"] * 100).round(1)
        fig_conv_trend = px.bar(
            conv_trend, x="period", y="conv_rate",
            color="period", color_discrete_sequence=PLOT_COLORS,
            text="conv_rate",
            labels={"conv_rate": "Consult → Purchase Rate (%)", "period": ""},
            title="Consult Conversion Rate YoY",
        )
        apply_dark_theme(fig_conv_trend)
        fig_conv_trend.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_conv_trend, use_container_width=True)

        # ── Sub-tabs: Part A (overall profile diff) vs Part B (YoY shift)
        s2a, s2b = st.tabs(["📊 Part A · Who Converts vs. Who Doesn't", "📅 Part B · How the Non-Converting Group Changed YoY"])

        # ════════════════════════════════════════════════════
        # PART A · PROFILE DIFFERENCES (existing analysis)
        # ════════════════════════════════════════════════════
        with s2a:
            st.markdown("#### What's Different Between Converters and Non-Converters?")
            st.caption("All periods combined. Looking for structural differences in who buys vs. who doesn't after a consult.")

            comp_cols = st.columns(2)

            with comp_cols[0]:
                if "student_grade_at_session_starts_at" in consult_pop.columns:
                    g_cp = consult_pop.copy()
                    g_cp["grade"] = pd.to_numeric(g_cp["student_grade_at_session_starts_at"], errors="coerce")
                    g_cp = g_cp.dropna(subset=["grade"])
                    g_cp["grade_label"] = "Gr " + g_cp["grade"].astype(int).astype(str)
                    g_mix = g_cp.groupby(["purchase_label", "grade_label"]).size().reset_index(name="count")
                    g_mix["pct"] = g_mix["count"] / g_mix.groupby("purchase_label")["count"].transform("sum") * 100
                    fig_g_cp = px.bar(
                        g_mix, x="grade_label", y="pct", color="purchase_label",
                        barmode="group", color_discrete_sequence=PLOT_COLORS,
                        labels={"pct": "% of Group", "grade_label": ""},
                        title="Grade Mix: Converted vs. Not",
                    )
                    apply_dark_theme(fig_g_cp)
                    st.plotly_chart(fig_g_cp, use_container_width=True)

            with comp_cols[1]:
                eb_mix = consult_pop.groupby(["purchase_label", "exam_bucket"]).size().reset_index(name="count")
                eb_mix["pct"] = eb_mix["count"] / eb_mix.groupby("purchase_label")["count"].transform("sum") * 100
                fig_eb_cp = px.bar(
                    eb_mix, x="exam_bucket", y="pct", color="purchase_label",
                    barmode="group", color_discrete_sequence=PLOT_COLORS,
                    labels={"pct": "% of Group", "exam_bucket": ""},
                    title="Exam Type: Converted vs. Not",
                )
                apply_dark_theme(fig_eb_cp)
                st.plotly_chart(fig_eb_cp, use_container_width=True)

            comp_cols2 = st.columns(2)
            with comp_cols2[0]:
                if "attended" in consult_pop.columns:
                    att_cp = (
                        consult_pop.groupby("purchase_label")
                        .agg(total=("attended", "count"), attended=("attended", "sum"))
                        .reset_index()
                    )
                    att_cp["att_rate"] = (att_cp["attended"] / att_cp["total"] * 100).round(1)
                    fig_att_cp = px.bar(
                        att_cp, x="purchase_label", y="att_rate",
                        color="purchase_label", color_discrete_sequence=PLOT_COLORS,
                        text="att_rate",
                        labels={"att_rate": "Attendance Rate (%)", "purchase_label": ""},
                        title="Attendance Rate: Converted vs. Not",
                    )
                    apply_dark_theme(fig_att_cp)
                    fig_att_cp.update_traces(texttemplate="%{text}%", textposition="outside")
                    st.plotly_chart(fig_att_cp, use_container_width=True)

            with comp_cols2[1]:
                if "days_to_first_consult" in consult_pop.columns:
                    dtc_cp = consult_pop[consult_pop["days_to_first_consult"].notna() & (consult_pop["days_to_first_consult"] >= 0)]
                    fig_dtc_cp = px.box(
                        dtc_cp, x="purchase_label", y="days_to_first_consult",
                        color="purchase_label", color_discrete_sequence=PLOT_COLORS,
                        labels={"days_to_first_consult": "Days to First Consult", "purchase_label": ""},
                        title="Days to First Consult: Converted vs. Not",
                    )
                    apply_dark_theme(fig_dtc_cp)
                    st.plotly_chart(fig_dtc_cp, use_container_width=True)

            comp_cols3 = st.columns(2)
            with comp_cols3[0]:
                if "ly_top_100" in consult_pop.columns:
                    top_cp = consult_pop.groupby(["purchase_label", "ly_top_100"]).size().reset_index(name="count")
                    top_cp = top_cp[top_cp["ly_top_100"].isin([0, 1])]
                    top_cp["tier"] = top_cp["ly_top_100"].map({1: "Top 100", 0: "Non-Top-100"})
                    top_cp["pct"] = top_cp["count"] / top_cp.groupby("purchase_label")["count"].transform("sum") * 100
                    fig_top_cp = px.bar(
                        top_cp, x="tier", y="pct", color="purchase_label",
                        barmode="group", color_discrete_sequence=PLOT_COLORS,
                        labels={"pct": "% of Group", "tier": ""},
                        title="Top 100 School Mix: Converted vs. Not",
                    )
                    apply_dark_theme(fig_top_cp)
                    st.plotly_chart(fig_top_cp, use_container_width=True)

            with comp_cols3[1]:
                if "enrollee_lead_source_region" in consult_pop.columns:
                    reg_cp = consult_pop.groupby(["purchase_label", "enrollee_lead_source_region"]).size().reset_index(name="count")
                    reg_cp["pct"] = reg_cp["count"] / reg_cp.groupby("purchase_label")["count"].transform("sum") * 100
                    fig_reg_cp = px.bar(
                        reg_cp, x="enrollee_lead_source_region", y="pct", color="purchase_label",
                        barmode="group", color_discrete_sequence=PLOT_COLORS,
                        labels={"pct": "% of Group", "enrollee_lead_source_region": ""},
                        title="Region Mix: Converted vs. Not",
                    )
                    apply_dark_theme(fig_reg_cp)
                    fig_reg_cp.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig_reg_cp, use_container_width=True)

            if "promotion_type" in consult_pop.columns:
                promo_cp = consult_pop.groupby(["purchase_label", "promotion_type"]).size().reset_index(name="count")
                promo_cp["pct"] = promo_cp["count"] / promo_cp.groupby("purchase_label")["count"].transform("sum") * 100
                fig_promo_cp = px.bar(
                    promo_cp, x="promotion_type", y="pct", color="purchase_label",
                    barmode="group", color_discrete_sequence=PLOT_COLORS,
                    labels={"pct": "% of Group", "promotion_type": ""},
                    title="Promotion Type Mix: Converted vs. Not",
                )
                apply_dark_theme(fig_promo_cp)
                fig_promo_cp.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_promo_cp, use_container_width=True)

            # Summary table
            st.markdown("#### Summary Comparison Table")
            rows_cp = []
            for lbl, grp in consult_pop.groupby("purchase_label"):
                n = len(grp)
                grade_avg = pd.to_numeric(grp.get("student_grade_at_session_starts_at", pd.Series(dtype=float)), errors="coerce").mean()
                att_r   = grp["attended"].sum() / n * 100 if "attended" in grp.columns and n else None
                top100_r = (grp["ly_top_100"] == 1).sum() / n * 100 if "ly_top_100" in grp.columns and n else None
                dtc_m   = grp["days_to_first_consult"].median() if "days_to_first_consult" in grp.columns else None
                rows_cp.append({
                    "Group": lbl, "N": n,
                    "Avg Grade": f"{grade_avg:.2f}" if not pd.isna(grade_avg) else "—",
                    "Attendance Rate": f"{round(att_r,1)}%" if att_r is not None else "—",
                    "Top 100 Share": f"{round(top100_r,1)}%" if top100_r is not None else "—",
                    "Median Days to Consult": f"{round(dtc_m,1)}" if dtc_m is not None and not pd.isna(dtc_m) else "—",
                })
            st.dataframe(pd.DataFrame(rows_cp).set_index("Group"), use_container_width=True)

        # ════════════════════════════════════════════════════
        # PART B · YoY SHIFT IN THE NON-CONVERTING GROUP
        # ════════════════════════════════════════════════════
        with s2b:
            st.markdown("#### How Has the Non-Converting Consult Population Changed from FY25 → FY26?")
            st.caption("Isolating only students who had a consult but did NOT purchase. Every chart below compares this group in FY25 vs. FY26 to spot what's shifted.")

            no_pur = consult_pop[consult_pop["purchased"] == 0].copy()
            did_pur = consult_pop[consult_pop["purchased"] == 1].copy()

            if no_pur.empty:
                st.warning("No non-converting consult records found.")
            else:
                # Size of the non-converting group per period
                nopur_size = no_pur.groupby("period").size().reset_index(name="count")
                didpur_size = did_pur.groupby("period").size().reset_index(name="count")

                yoy_c1, yoy_c2 = st.columns(2)
                with yoy_c1:
                    nopur_size["label"] = "❌ Did Not Purchase"
                    didpur_size["label"] = "✅ Purchased"
                    size_combined = pd.concat([nopur_size, didpur_size])
                    fig_size = px.bar(
                        size_combined, x="period", y="count", color="label",
                        barmode="stack", color_discrete_sequence=[PLOT_COLORS[1], PLOT_COLORS[0]],
                        labels={"count": "# with Consult", "period": ""},
                        title="Consult Population Size by Outcome & Period",
                    )
                    apply_dark_theme(fig_size)
                    st.plotly_chart(fig_size, use_container_width=True)

                with yoy_c2:
                    # Non-converting group as % of total consult pop
                    nopur_pct = conv_trend.copy()
                    nopur_pct["nopur_rate"] = 100 - nopur_pct["conv_rate"]
                    fig_nopur_pct = px.bar(
                        nopur_pct, x="period", y="nopur_rate",
                        color="period", color_discrete_sequence=PLOT_COLORS,
                        text="nopur_rate",
                        labels={"nopur_rate": "Non-Convert Rate (%)", "period": ""},
                        title="% of Consult Population That Did NOT Purchase",
                    )
                    apply_dark_theme(fig_nopur_pct)
                    fig_nopur_pct.update_traces(texttemplate="%{text}%", textposition="outside")
                    st.plotly_chart(fig_nopur_pct, use_container_width=True)

                st.markdown("---")
                st.markdown("##### Profile of the Non-Converting Group: FY25 vs. FY26")

                yoy_r1c1, yoy_r1c2 = st.columns(2)

                # Grade mix YoY — non-converters only
                with yoy_r1c1:
                    if "student_grade_at_session_starts_at" in no_pur.columns:
                        np_g = no_pur.copy()
                        np_g["grade"] = pd.to_numeric(np_g["student_grade_at_session_starts_at"], errors="coerce")
                        np_g = np_g.dropna(subset=["grade"])
                        np_g["grade_label"] = "Gr " + np_g["grade"].astype(int).astype(str)
                        np_gmix = np_g.groupby(["period", "grade_label"]).size().reset_index(name="count")
                        np_gmix["pct"] = np_gmix["count"] / np_gmix.groupby("period")["count"].transform("sum") * 100
                        fig_np_g = px.bar(
                            np_gmix, x="grade_label", y="pct", color="period",
                            barmode="group", color_discrete_sequence=PLOT_COLORS,
                            labels={"pct": "% of Non-Converters", "grade_label": ""},
                            title="Grade Mix of Non-Converters: FY25 vs. FY26",
                        )
                        apply_dark_theme(fig_np_g)
                        st.plotly_chart(fig_np_g, use_container_width=True)

                # Exam type mix YoY — non-converters only
                with yoy_r1c2:
                    np_emix = no_pur.groupby(["period", "exam_bucket"]).size().reset_index(name="count")
                    np_emix["pct"] = np_emix["count"] / np_emix.groupby("period")["count"].transform("sum") * 100
                    fig_np_e = px.bar(
                        np_emix, x="exam_bucket", y="pct", color="period",
                        barmode="group", color_discrete_sequence=PLOT_COLORS,
                        labels={"pct": "% of Non-Converters", "exam_bucket": ""},
                        title="Exam Type Mix of Non-Converters: FY25 vs. FY26",
                    )
                    apply_dark_theme(fig_np_e)
                    st.plotly_chart(fig_np_e, use_container_width=True)

                yoy_r2c1, yoy_r2c2 = st.columns(2)

                # Days to consult YoY — non-converters only
                with yoy_r2c1:
                    if "days_to_first_consult" in no_pur.columns:
                        np_dtc = no_pur[no_pur["days_to_first_consult"].notna() & (no_pur["days_to_first_consult"] >= 0)]
                        fig_np_dtc = px.box(
                            np_dtc, x="period", y="days_to_first_consult",
                            color="period", color_discrete_sequence=PLOT_COLORS,
                            labels={"days_to_first_consult": "Days to First Consult", "period": ""},
                            title="Days to Consult — Non-Converters FY25 vs. FY26",
                        )
                        apply_dark_theme(fig_np_dtc)
                        st.plotly_chart(fig_np_dtc, use_container_width=True)

                # Attendance rate YoY — non-converters only
                with yoy_r2c2:
                    if "attended" in no_pur.columns:
                        np_att = (
                            no_pur.groupby("period")
                            .agg(total=("attended", "count"), attended=("attended", "sum"))
                            .reset_index()
                        )
                        np_att["att_rate"] = (np_att["attended"] / np_att["total"] * 100).round(1)
                        fig_np_att = px.bar(
                            np_att, x="period", y="att_rate",
                            color="period", color_discrete_sequence=PLOT_COLORS,
                            text="att_rate",
                            labels={"att_rate": "Attendance Rate (%)", "period": ""},
                            title="Attendance Rate — Non-Converters FY25 vs. FY26",
                        )
                        apply_dark_theme(fig_np_att)
                        fig_np_att.update_traces(texttemplate="%{text}%", textposition="outside")
                        st.plotly_chart(fig_np_att, use_container_width=True)

                yoy_r3c1, yoy_r3c2 = st.columns(2)

                # Top 100 share YoY — non-converters only
                with yoy_r3c1:
                    if "ly_top_100" in no_pur.columns:
                        np_top = (
                            no_pur.groupby("period")
                            .agg(total=("enrollment_id", "count"), top100=("ly_top_100", "sum"))
                            .reset_index()
                        )
                        np_top["top100_pct"] = (np_top["top100"] / np_top["total"] * 100).round(1)
                        fig_np_top = px.bar(
                            np_top, x="period", y="top100_pct",
                            color="period", color_discrete_sequence=PLOT_COLORS,
                            text="top100_pct",
                            labels={"top100_pct": "% Top 100 School", "period": ""},
                            title="Top 100 Share — Non-Converters FY25 vs. FY26",
                        )
                        apply_dark_theme(fig_np_top)
                        fig_np_top.update_traces(texttemplate="%{text}%", textposition="outside")
                        st.plotly_chart(fig_np_top, use_container_width=True)

                # Region mix YoY — non-converters only
                with yoy_r3c2:
                    if "enrollee_lead_source_region" in no_pur.columns:
                        np_reg = no_pur.groupby(["period", "enrollee_lead_source_region"]).size().reset_index(name="count")
                        np_reg["pct"] = np_reg["count"] / np_reg.groupby("period")["count"].transform("sum") * 100
                        fig_np_reg = px.bar(
                            np_reg, x="enrollee_lead_source_region", y="pct", color="period",
                            barmode="group", color_discrete_sequence=PLOT_COLORS,
                            labels={"pct": "% of Non-Converters", "enrollee_lead_source_region": ""},
                            title="Region Mix — Non-Converters FY25 vs. FY26",
                        )
                        apply_dark_theme(fig_np_reg)
                        fig_np_reg.update_layout(xaxis_tickangle=-35)
                        st.plotly_chart(fig_np_reg, use_container_width=True)

                # Promotion type mix YoY — non-converters only
                if "promotion_type" in no_pur.columns:
                    np_promo = no_pur.groupby(["period", "promotion_type"]).size().reset_index(name="count")
                    np_promo["pct"] = np_promo["count"] / np_promo.groupby("period")["count"].transform("sum") * 100
                    fig_np_promo = px.bar(
                        np_promo, x="promotion_type", y="pct", color="period",
                        barmode="group", color_discrete_sequence=PLOT_COLORS,
                        labels={"pct": "% of Non-Converters", "promotion_type": ""},
                        title="Promotion Type — Non-Converters FY25 vs. FY26",
                    )
                    apply_dark_theme(fig_np_promo)
                    fig_np_promo.update_layout(xaxis_tickangle=-30)
                    st.plotly_chart(fig_np_promo, use_container_width=True)

                # YoY summary scorecard for non-converters
                st.markdown("##### Non-Converter Scorecard: FY25 vs. FY26")
                np_rows = []
                for p, grp in no_pur.groupby("period"):
                    n = len(grp)
                    grade_avg = pd.to_numeric(grp.get("student_grade_at_session_starts_at", pd.Series(dtype=float)), errors="coerce").mean()
                    att_r    = grp["attended"].sum() / n * 100 if "attended" in grp.columns and n else None
                    top100_r = (grp["ly_top_100"] == 1).sum() / n * 100 if "ly_top_100" in grp.columns and n else None
                    dtc_m    = grp["days_to_first_consult"].median() if "days_to_first_consult" in grp.columns else None
                    unprom   = grp["promotion_type"].astype(str).str.contains("Unpromoted", na=False).sum() / n * 100 if "promotion_type" in grp.columns and n else None
                    np_rows.append({
                        "Period": p,
                        "Non-Converters (N)": n,
                        "Avg Grade": f"{grade_avg:.2f}" if not pd.isna(grade_avg) else "—",
                        "Attendance Rate": f"{round(att_r,1)}%" if att_r is not None else "—",
                        "Top 100 Share": f"{round(top100_r,1)}%" if top100_r is not None else "—",
                        "Median Days to Consult": f"{round(dtc_m,1)}" if dtc_m is not None and not pd.isna(dtc_m) else "—",
                        "Unpromoted Share": f"{round(unprom,1)}%" if unprom is not None else "—",
                    })
                st.dataframe(pd.DataFrame(np_rows).set_index("Period"), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 · ATLANTIC / MIDWEST / SOUTH REGIONAL DROP
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 3 · Atlantic / Midwest / South — What's Behind the Sales Drop?")
    st.caption("Isolating these three regions to identify which factors diverged most between periods.")

    FOCUS_REGIONS = ["Atlantic", "Midwest", "Midsouth", "South", "Southeast",
                     "Northeast Metro", "Great Lakes States"]

    # Let user adjust which regions to include in this section
    if "enrollee_lead_source_region" in df.columns:
        all_reg_opts = sorted(df["enrollee_lead_source_region"].dropna().unique().tolist())
        focus_regions_sel = st.multiselect(
            "Regions to investigate (pre-loaded with Atlantic/Midwest/South variants found in data)",
            options=all_reg_opts,
            default=[r for r in all_reg_opts if any(k.lower() in r.lower() for k in
                     ["atlantic", "midwest", "midsouth", "south", "great lakes"])],
            key="focus_region_sel",
        )
    else:
        focus_regions_sel = []

    focus_df = df[df["enrollee_lead_source_region"].isin(focus_regions_sel)].copy() if focus_regions_sel else pd.DataFrame()

    if focus_df.empty:
        st.warning("No data for selected focus regions after current filters.")
    else:
        # ── 3a: Volume & close rate side-by-side
        st.markdown("#### Enrollment Volume & Close Rate by Region & Period")
        reg3_agg = (
            focus_df.groupby(["period", "enrollee_lead_source_region"])
            .agg(
                enrollments=("enrollment_id", "count"),
                purchases=("new_purchase_flag", "sum"),
                attended=("attended", "sum"),
                consults=("completed_consult_flag", "sum"),
            )
            .reset_index()
        )
        reg3_agg["close_rate"] = (reg3_agg["purchases"] / reg3_agg["enrollments"] * 100).round(1)
        reg3_agg["att_rate"] = (reg3_agg["attended"] / reg3_agg["enrollments"] * 100).round(1)
        reg3_agg["consult_rate"] = (reg3_agg["consults"] / reg3_agg["enrollments"] * 100).round(1)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            fig_r3_vol = px.bar(
                reg3_agg, x="enrollee_lead_source_region", y="enrollments", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"enrollments": "# Enrollments", "enrollee_lead_source_region": ""},
                title="Enrollment Volume by Region & Period",
            )
            apply_dark_theme(fig_r3_vol)
            fig_r3_vol.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_r3_vol, use_container_width=True)

        with r3c2:
            fig_r3_cr = px.bar(
                reg3_agg, x="enrollee_lead_source_region", y="close_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                text="close_rate",
                labels={"close_rate": "Close Rate (%)", "enrollee_lead_source_region": ""},
                title="Close Rate by Region & Period",
            )
            apply_dark_theme(fig_r3_cr)
            fig_r3_cr.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_r3_cr.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_r3_cr, use_container_width=True)

        # ── 3b: Waterfall — what's changed
        st.markdown("#### Key Metrics Comparison: Focus Regions YoY")
        r3c3, r3c4 = st.columns(2)

        with r3c3:
            fig_r3_att = px.bar(
                reg3_agg, x="enrollee_lead_source_region", y="att_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"att_rate": "Attendance Rate (%)", "enrollee_lead_source_region": ""},
                title="Attendance Rate by Region & Period",
            )
            apply_dark_theme(fig_r3_att)
            fig_r3_att.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_r3_att, use_container_width=True)

        with r3c4:
            fig_r3_cons = px.bar(
                reg3_agg, x="enrollee_lead_source_region", y="consult_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"consult_rate": "Consult Completion Rate (%)", "enrollee_lead_source_region": ""},
                title="Consult Rate by Region & Period",
            )
            apply_dark_theme(fig_r3_cons)
            fig_r3_cons.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_r3_cons, use_container_width=True)

        # ── 3c: Exam type mix shift within focus regions
        st.markdown("#### Exam Type Mix Shift in Focus Regions")
        st.caption("Did these regions see a shift toward lower-converting exam types?")
        exam_reg3 = (
            focus_df.groupby(["period", "exam_bucket"])
            .agg(enrollments=("enrollment_id", "count"), purchases=("new_purchase_flag", "sum"))
            .reset_index()
        )
        exam_reg3["close_rate"] = (exam_reg3["purchases"] / exam_reg3["enrollments"] * 100).round(1)
        exam_reg3["pct_of_period"] = exam_reg3["enrollments"] / exam_reg3.groupby("period")["enrollments"].transform("sum") * 100

        r3c5, r3c6 = st.columns(2)
        with r3c5:
            fig_r3_ex_mix = px.bar(
                exam_reg3, x="exam_bucket", y="pct_of_period", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"pct_of_period": "% of Enrollments", "exam_bucket": ""},
                title="Exam Mix in Focus Regions (% share)",
            )
            apply_dark_theme(fig_r3_ex_mix)
            st.plotly_chart(fig_r3_ex_mix, use_container_width=True)

        with r3c6:
            fig_r3_ex_cr = px.bar(
                exam_reg3, x="exam_bucket", y="close_rate", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                text="close_rate",
                labels={"close_rate": "Close Rate (%)", "exam_bucket": ""},
                title="Close Rate by Exam Type — Focus Regions Only",
            )
            apply_dark_theme(fig_r3_ex_cr)
            fig_r3_ex_cr.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_r3_ex_cr, use_container_width=True)

        # ── 3d: Grade mix shift within focus regions
        st.markdown("#### Grade Mix Shift in Focus Regions")
        if "student_grade_at_session_starts_at" in focus_df.columns:
            gr_reg3 = focus_df.copy()
            gr_reg3["grade"] = pd.to_numeric(gr_reg3["student_grade_at_session_starts_at"], errors="coerce")
            gr_reg3 = gr_reg3.dropna(subset=["grade"])
            gr_reg3["grade_label"] = "Gr " + gr_reg3["grade"].astype(int).astype(str)
            gr_mix3 = gr_reg3.groupby(["period", "grade_label"]).size().reset_index(name="count")
            gr_mix3["pct"] = gr_mix3["count"] / gr_mix3.groupby("period")["count"].transform("sum") * 100
            fig_gr3 = px.bar(
                gr_mix3, x="grade_label", y="pct", color="period",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"pct": "% of Enrollments", "grade_label": "Grade"},
                title="Grade Mix in Focus Regions YoY",
            )
            apply_dark_theme(fig_gr3)
            st.plotly_chart(fig_gr3, use_container_width=True)

        # ── 3e: Promotion type mix shift
        if "promotion_type" in focus_df.columns:
            st.markdown("#### Promotion Type Mix in Focus Regions")
            promo_reg3 = focus_df.groupby(["period", "promotion_type"]).size().reset_index(name="count")
            promo_reg3["pct"] = promo_reg3["count"] / promo_reg3.groupby("period")["count"].transform("sum") * 100
            fig_pr3 = px.bar(
                promo_reg3, x="period", y="pct", color="promotion_type",
                barmode="stack", color_discrete_sequence=PLOT_COLORS,
                labels={"pct": "% of Enrollments", "promotion_type": "Type"},
                title="Promotion Mix in Focus Regions (stacked %)",
            )
            apply_dark_theme(fig_pr3)
            st.plotly_chart(fig_pr3, use_container_width=True)

        # ── 3f: Top 100 school share in focus regions
        if "ly_top_100" in focus_df.columns:
            st.markdown("#### Top 100 School Share in Focus Regions")
            top_reg3 = (
                focus_df.groupby("period")
                .agg(total=("enrollment_id", "count"), top100=("ly_top_100", "sum"))
                .reset_index()
            )
            top_reg3["top100_pct"] = (top_reg3["top100"] / top_reg3["total"] * 100).round(1)
            fig_top3 = px.bar(
                top_reg3, x="period", y="top100_pct",
                color="period", color_discrete_sequence=PLOT_COLORS,
                text="top100_pct",
                labels={"top100_pct": "% Top 100 School Enrollments", "period": ""},
                title="Top 100 School Share — Focus Regions YoY",
            )
            apply_dark_theme(fig_top3)
            fig_top3.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_top3, use_container_width=True)

        # ── 3g: Advisor performance in focus regions
        if "enrollee_lead_source_advisor_name" in focus_df.columns:
            st.markdown("#### Advisor Performance in Focus Regions")
            adv_reg3 = (
                focus_df.groupby(["period", "enrollee_lead_source_advisor_name"])
                .agg(
                    enrollments=("enrollment_id", "count"),
                    purchases=("new_purchase_flag", "sum"),
                    consults=("completed_consult_flag", "sum"),
                )
                .reset_index()
            )
            adv_reg3["close_rate"] = (adv_reg3["purchases"] / adv_reg3["enrollments"] * 100).round(1)
            adv_reg3["consult_rate"] = (adv_reg3["consults"] / adv_reg3["enrollments"] * 100).round(1)
            adv_reg3 = adv_reg3.sort_values(["period", "enrollments"], ascending=[True, False])

            for p in sorted(adv_reg3["period"].dropna().unique()):
                with st.expander(f"📋 {p} — Advisor Close Rates in Focus Regions"):
                    st.dataframe(
                        adv_reg3[adv_reg3["period"] == p]
                        .drop("period", axis=1)
                        .head(25)
                        .set_index("enrollee_lead_source_advisor_name"),
                        use_container_width=True,
                    )

        # ── 3h: Summary delta table for focus regions
        st.markdown("#### Focus Region Summary: FY25 vs. FY26")
        focus_summary = []
        for p, grp in focus_df.groupby("period"):
            n = len(grp)
            cr = grp["new_purchase_flag"].sum() / n * 100 if n else None
            ar = grp["attended"].sum() / n * 100 if "attended" in grp.columns and n else None
            conr = grp["completed_consult_flag"].sum() / n * 100 if "completed_consult_flag" in grp.columns and n else None
            t100 = (grp["ly_top_100"] == 1).sum() / n * 100 if "ly_top_100" in grp.columns and n else None
            dtc = grp["days_to_first_consult"].median() if "days_to_first_consult" in grp.columns else None
            focus_summary.append({
                "Period": p,
                "Enrollments": n,
                "Close Rate %": f"{round(cr,1)}%" if cr is not None else "—",
                "Attendance Rate %": f"{round(ar,1)}%" if ar is not None else "—",
                "Consult Rate %": f"{round(conr,1)}%" if conr is not None else "—",
                "Top 100 Share %": f"{round(t100,1)}%" if t100 is not None else "—",
                "Median Days to Consult": f"{round(dtc,1)}" if dtc is not None and not pd.isna(dtc) else "—",
            })
        st.dataframe(pd.DataFrame(focus_summary).set_index("Period"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 · TIMING & CADENCE  (Jan 2024 → Feb 2026, full window)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown("## ⏱️ Timing & Cadence Analysis")
    st.markdown("""
    <div class="highlight-box">
    Uses the <b>full dataset (Jan 2024 – Feb 2026)</b> — not limited to the two Oct–Feb comparison windows.
    Sidebar subject / region / event-type filters still apply. Grade and region breakdowns are available
    inside each section via inline controls.
    </div>
    """, unsafe_allow_html=True)

    # ── Build full-window dataset ─────────────────────────────────────────────
    @st.cache_data
    def build_timing_df(raw_df):
        tdf = raw_df[
            (raw_df["session_starts_at"] >= "2024-01-01") &
            (raw_df["session_starts_at"] <= "2026-02-28")
        ].copy()

        # Exam bucket
        def _eb(s):
            s = str(s).upper()
            if "ACT" in s:            return "ACT"
            if "PSAT" in s or "NMSQT" in s: return "PSAT/NMSQT"
            if "SAT" in s:            return "SAT"
            if "AP " in s or s.startswith("AP"): return "AP"
            return "Other"

        tdf["exam_bucket"] = tdf["course_subject_name"].apply(_eb)

        # Calendar fields
        tdf["exam_month"]     = tdf["session_starts_at"].dt.to_period("M").astype(str)
        tdf["exam_month_num"] = tdf["session_starts_at"].dt.month
        tdf["exam_year"]      = tdf["session_starts_at"].dt.year
        tdf["exam_ym_dt"]     = tdf["session_starts_at"].dt.to_period("M").dt.to_timestamp()

        # Numeric fields
        tdf["days_exam_to_booking"] = pd.to_numeric(tdf["days_to_first_booking"], errors="coerce")
        tdf["days_exam_to_consult"] = pd.to_numeric(tdf["days_to_first_consult"],  errors="coerce")
        tdf["purchased"]            = pd.to_numeric(tdf["new_purchase_flag"],       errors="coerce").fillna(0).astype(int)
        tdf["consulted"]            = pd.to_numeric(tdf["completed_consult_flag"],  errors="coerce").fillna(0).astype(int)
        tdf["grade_num"]            = pd.to_numeric(tdf["student_grade_at_session_starts_at"], errors="coerce")

        return tdf

    tdf_full = build_timing_df(raw)

    # Apply sidebar subject / region / event-type filters to timing df too
    tdf = tdf_full.copy()
    if selected_event_types:
        tdf = tdf[tdf["promotion_type"].isin(selected_event_types)]
    if selected_subjects:
        tdf = tdf[tdf["course_subject_name"].isin(selected_subjects)]
    if selected_regions:
        tdf = tdf[tdf["enrollee_lead_source_region"].isin(selected_regions)]

    ACT_SAT_PSAT = ["ACT", "SAT", "PSAT/NMSQT"]

    # ── Shared inline filter controls ────────────────────────────────────────
    st.markdown("---")
    tc_filt_col1, tc_filt_col2, tc_filt_col3 = st.columns(3)
    with tc_filt_col1:
        tc_exam_types = st.multiselect(
            "Exam Types (all sections)",
            options=ACT_SAT_PSAT + ["AP", "Other"],
            default=ACT_SAT_PSAT,
            key="tc_exam_filter",
        )
    with tc_filt_col2:
        grade_opts = sorted(tdf["grade_num"].dropna().unique().astype(int).tolist())
        tc_grades = st.multiselect(
            "Grade(s) at Exam Time",
            options=grade_opts,
            default=grade_opts,
            key="tc_grade_filter",
            format_func=lambda g: f"Grade {g}",
        )
    with tc_filt_col3:
        reg_opts_tc = sorted(tdf["enrollee_lead_source_region"].dropna().unique().tolist())
        tc_regions = st.multiselect(
            "Region(s)",
            options=reg_opts_tc,
            default=reg_opts_tc,
            key="tc_region_filter",
        )

    # Apply inline filters
    tdf_f = tdf[
        tdf["exam_bucket"].isin(tc_exam_types) &
        tdf["grade_num"].isin(tc_grades) &
        tdf["enrollee_lead_source_region"].isin(tc_regions)
    ].copy() if tc_grades and tc_regions and tc_exam_types else tdf.copy()

    if tdf_f.empty:
        st.warning("No data after filters — try broadening the selections above.")
        st.stop()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 · EXAM DATE → CLOSE  (days_to_first_booking)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 1 · Time from Exam to Close — Shifting Over Time?")
    st.caption("Median days from mock exam session to first booking, trended by month. A rising line = leads taking longer to convert.")

    etb = tdf_f[tdf_f["days_exam_to_booking"].notna() & (tdf_f["days_exam_to_booking"] > 0)].copy()

    if not etb.empty:
        # ── 1a: Median days exam→close trended monthly, colored by exam type
        etb_trend = (
            etb.groupby(["exam_ym_dt", "exam_bucket"])["days_exam_to_booking"]
            .median().reset_index()
            .rename(columns={"days_exam_to_booking": "median_days"})
            .sort_values("exam_ym_dt")
        )
        fig_etb_trend = px.line(
            etb_trend, x="exam_ym_dt", y="median_days", color="exam_bucket",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"median_days": "Median Days to Close", "exam_ym_dt": "Exam Month", "exam_bucket": "Exam Type"},
            title="Median Days: Exam → Close by Month & Exam Type",
        )
        apply_dark_theme(fig_etb_trend)
        fig_etb_trend.update_layout(xaxis_tickformat="%b %Y")
        st.plotly_chart(fig_etb_trend, use_container_width=True)

        tc1c1, tc1c2 = st.columns(2)

        # ── 1b: Distribution shift — box plot by year
        with tc1c1:
            etb["exam_year_str"] = etb["exam_year"].astype(str)
            fig_etb_box = px.box(
                etb, x="exam_bucket", y="days_exam_to_booking", color="exam_year_str",
                color_discrete_sequence=PLOT_COLORS,
                labels={"days_exam_to_booking": "Days to Close", "exam_bucket": "Exam Type", "exam_year_str": "Year"},
                title="Distribution: Days Exam → Close by Exam Type & Year",
            )
            apply_dark_theme(fig_etb_box)
            st.plotly_chart(fig_etb_box, use_container_width=True)

        # ── 1c: Breakdown by grade
        with tc1c2:
            etb_grade = (
                etb[etb["grade_num"].notna()]
                .groupby(["grade_num", "exam_bucket"])["days_exam_to_booking"]
                .median().reset_index()
                .rename(columns={"days_exam_to_booking": "median_days"})
            )
            etb_grade["grade_label"] = "Gr " + etb_grade["grade_num"].astype(int).astype(str)
            fig_etb_grade = px.bar(
                etb_grade, x="grade_label", y="median_days", color="exam_bucket",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"median_days": "Median Days to Close", "grade_label": "Grade", "exam_bucket": "Exam Type"},
                title="Median Days Exam → Close by Grade & Exam Type",
            )
            apply_dark_theme(fig_etb_grade)
            st.plotly_chart(fig_etb_grade, use_container_width=True)

        # ── 1d: By region
        if "enrollee_lead_source_region" in etb.columns:
            etb_reg = (
                etb.groupby(["enrollee_lead_source_region", "exam_bucket"])["days_exam_to_booking"]
                .median().reset_index()
                .rename(columns={"days_exam_to_booking": "median_days"})
            )
            fig_etb_reg = px.bar(
                etb_reg, x="enrollee_lead_source_region", y="median_days", color="exam_bucket",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"median_days": "Median Days to Close", "enrollee_lead_source_region": "", "exam_bucket": "Exam Type"},
                title="Median Days Exam → Close by Region & Exam Type",
            )
            apply_dark_theme(fig_etb_reg)
            fig_etb_reg.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_etb_reg, use_container_width=True)

        # ── 1e: Speed buckets over time
        etb["close_speed"] = pd.cut(
            etb["days_exam_to_booking"],
            bins=[-1, 30, 60, 90, 180, 99999],
            labels=["0–30 days", "31–60 days", "61–90 days", "91–180 days", "180+ days"],
        )
        speed_trend = (
            etb.groupby(["exam_ym_dt", "close_speed"])
            .size().reset_index(name="count")
            .sort_values("exam_ym_dt")
        )
        fig_speed_trend = px.bar(
            speed_trend, x="exam_ym_dt", y="count", color="close_speed",
            barmode="stack", color_discrete_sequence=PLOT_COLORS,
            labels={"count": "Closes", "exam_ym_dt": "Exam Month", "close_speed": "Time to Close"},
            title="Close Speed Buckets Over Time (Exam → Close)",
        )
        apply_dark_theme(fig_speed_trend)
        fig_speed_trend.update_layout(xaxis_tickformat="%b %Y")
        st.plotly_chart(fig_speed_trend, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 · CONSULT DATE → CLOSE  (days_to_first_booking relative timing)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 2 · Time from Consult to Close — Shifting Over Time?")
    st.caption("Derived as days_to_first_booking minus days_to_first_consult (proxy for consult→close lag). Trended monthly.")

    ctb = tdf_f[
        tdf_f["days_exam_to_booking"].notna() &
        tdf_f["days_exam_to_consult"].notna() &
        (tdf_f["days_exam_to_booking"] > 0) &
        (tdf_f["days_exam_to_consult"] >= 0)
    ].copy()
    ctb["days_consult_to_close"] = ctb["days_exam_to_booking"] - ctb["days_exam_to_consult"]
    ctb = ctb[ctb["days_consult_to_close"] >= 0]  # keep only logical values

    if not ctb.empty:
        ctb_trend = (
            ctb.groupby(["exam_ym_dt", "exam_bucket"])["days_consult_to_close"]
            .median().reset_index()
            .rename(columns={"days_consult_to_close": "median_days"})
            .sort_values("exam_ym_dt")
        )
        fig_ctb_trend = px.line(
            ctb_trend, x="exam_ym_dt", y="median_days", color="exam_bucket",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"median_days": "Median Days Consult → Close", "exam_ym_dt": "Exam Month", "exam_bucket": "Exam Type"},
            title="Median Days: Consult → Close by Month & Exam Type",
        )
        apply_dark_theme(fig_ctb_trend)
        fig_ctb_trend.update_layout(xaxis_tickformat="%b %Y")
        st.plotly_chart(fig_ctb_trend, use_container_width=True)

        tc2c1, tc2c2 = st.columns(2)

        with tc2c1:
            ctb["exam_year_str"] = ctb["exam_year"].astype(str)
            fig_ctb_box = px.box(
                ctb, x="exam_bucket", y="days_consult_to_close", color="exam_year_str",
                color_discrete_sequence=PLOT_COLORS,
                labels={"days_consult_to_close": "Days Consult → Close", "exam_bucket": "Exam Type", "exam_year_str": "Year"},
                title="Distribution: Days Consult → Close by Exam Type & Year",
            )
            apply_dark_theme(fig_ctb_box)
            st.plotly_chart(fig_ctb_box, use_container_width=True)

        with tc2c2:
            ctb_grade = (
                ctb[ctb["grade_num"].notna()]
                .groupby(["grade_num", "exam_bucket"])["days_consult_to_close"]
                .median().reset_index()
                .rename(columns={"days_consult_to_close": "median_days"})
            )
            ctb_grade["grade_label"] = "Gr " + ctb_grade["grade_num"].astype(int).astype(str)
            fig_ctb_grade = px.bar(
                ctb_grade, x="grade_label", y="median_days", color="exam_bucket",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"median_days": "Median Days", "grade_label": "Grade", "exam_bucket": "Exam Type"},
                title="Median Days Consult → Close by Grade & Exam Type",
            )
            apply_dark_theme(fig_ctb_grade)
            st.plotly_chart(fig_ctb_grade, use_container_width=True)

        if "enrollee_lead_source_region" in ctb.columns:
            ctb_reg = (
                ctb.groupby(["enrollee_lead_source_region", "exam_bucket"])["days_consult_to_close"]
                .median().reset_index()
                .rename(columns={"days_consult_to_close": "median_days"})
            )
            fig_ctb_reg = px.bar(
                ctb_reg, x="enrollee_lead_source_region", y="median_days", color="exam_bucket",
                barmode="group", color_discrete_sequence=PLOT_COLORS,
                labels={"median_days": "Median Days", "enrollee_lead_source_region": "", "exam_bucket": "Exam Type"},
                title="Median Days Consult → Close by Region & Exam Type",
            )
            apply_dark_theme(fig_ctb_reg)
            fig_ctb_reg.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_ctb_reg, use_container_width=True)

        # Exam→consult vs consult→close: stacked comparison
        st.markdown("#### Exam → Consult vs. Consult → Close: Where Is Time Being Lost?")
        st.caption("Stacked bars show how the total days-to-close is split between the two legs of the funnel.")
        ctb_split = (
            ctb.groupby(["exam_ym_dt", "exam_bucket"])
            .agg(
                exam_to_consult=("days_exam_to_consult", "median"),
                consult_to_close=("days_consult_to_close", "median"),
            )
            .reset_index()
            .sort_values("exam_ym_dt")
        )
        ctb_split_m = ctb_split.melt(
            id_vars=["exam_ym_dt", "exam_bucket"],
            value_vars=["exam_to_consult", "consult_to_close"],
            var_name="leg", value_name="median_days",
        )
        ctb_split_m["leg"] = ctb_split_m["leg"].map({
            "exam_to_consult": "Exam → Consult",
            "consult_to_close": "Consult → Close",
        })
        fig_ctb_split = px.bar(
            ctb_split_m, x="exam_ym_dt", y="median_days", color="leg",
            facet_col="exam_bucket", barmode="stack",
            color_discrete_sequence=[PLOT_COLORS[2], PLOT_COLORS[1]],
            labels={"median_days": "Median Days", "exam_ym_dt": "Exam Month", "leg": "Funnel Leg"},
            title="Funnel Split: Exam→Consult vs. Consult→Close Over Time",
        )
        apply_dark_theme(fig_ctb_split)
        fig_ctb_split.update_layout(xaxis_tickformat="%b %Y", xaxis_tickangle=-45)
        fig_ctb_split.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_ctb_split, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 · EXAM TIMING × CLOSE EFFECTIVENESS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 3 · Does Exam Timing Affect Close Effectiveness?")
    st.caption("Close rate by the calendar month the exam was held, split by exam type. Are certain months more predictive of conversion than others?")

    eff_df = tdf_f.copy()
    eff_df["month_order"] = eff_df["session_starts_at"].dt.month
    MONTH_SORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    MONTH_TICKVALS_ALL = list(range(1, 13))
    MONTH_TICKTEXT_ALL = MONTH_SORT

    def fix_month_axis_all(fig):
        fig.update_xaxes(tickmode="array", tickvals=MONTH_TICKVALS_ALL, ticktext=MONTH_TICKTEXT_ALL)
        return fig

    eff_agg = (
        eff_df[eff_df["exam_bucket"].isin(ACT_SAT_PSAT)]
        .groupby(["month_order", "exam_bucket"])
        .agg(enrollments=("enrollment_id", "count"), purchases=("purchased", "sum"), consults=("consulted", "sum"))
        .reset_index()
        .sort_values("month_order")
    )
    eff_agg["close_rate"]   = (eff_agg["purchases"] / eff_agg["enrollments"] * 100).round(1)
    eff_agg["consult_rate"] = (eff_agg["consults"]  / eff_agg["enrollments"] * 100).round(1)

    tc3c1, tc3c2 = st.columns(2)
    with tc3c1:
        fig_eff_cr = px.line(
            eff_agg, x="month_order", y="close_rate", color="exam_bucket",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"close_rate": "Close Rate (%)", "month_order": "Exam Month", "exam_bucket": "Exam Type"},
            title="Close Rate by Exam Month & Type (all years combined)",
        )
        apply_dark_theme(fig_eff_cr)
        fix_month_axis_all(fig_eff_cr)
        st.plotly_chart(fig_eff_cr, use_container_width=True)

    with tc3c2:
        fig_eff_con = px.line(
            eff_agg, x="month_order", y="consult_rate", color="exam_bucket",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"consult_rate": "Consult Rate (%)", "month_order": "Exam Month", "exam_bucket": "Exam Type"},
            title="Consult Rate by Exam Month & Type (all years combined)",
        )
        apply_dark_theme(fig_eff_con)
        fix_month_axis_all(fig_eff_con)
        st.plotly_chart(fig_eff_con, use_container_width=True)

    # Heatmap: close rate by month × exam type
    st.markdown("#### Heatmap: Close Rate by Month × Exam Type")
    heat_eff = eff_agg.pivot_table(
        index="exam_bucket", columns="month_order", values="close_rate", aggfunc="mean"
    )
    # Rename columns from int to month abbrevs for the heatmap labels
    heat_eff.columns = [MONTH_SORT[c-1] for c in heat_eff.columns]
    fig_eff_heat = px.imshow(
        heat_eff, text_auto=".1f",
        color_continuous_scale="Blues",
        labels={"color": "Close Rate (%)"},
        title="Close Rate Heatmap: Exam Month × Exam Type",
        aspect="auto",
    )
    fig_eff_heat.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans, sans-serif", color=PLOT_FONT, size=11),
        margin=dict(t=50, b=30, l=10, r=10),
        coloraxis_colorbar=dict(tickfont_color=PLOT_FONT, title_font_color=PLOT_FONT),
    )
    st.plotly_chart(fig_eff_heat, use_container_width=True)

    # By grade: close rate per exam month, faceted
    st.markdown("#### Close Rate by Exam Month & Grade (per Exam Type)")
    st.caption("Use this to see whether certain grade levels respond differently to specific testing months.")
    eff_grade = (
        eff_df[eff_df["exam_bucket"].isin(ACT_SAT_PSAT) & eff_df["grade_num"].notna()]
        .groupby(["month_order", "exam_bucket", "grade_num"])
        .agg(enrollments=("enrollment_id", "count"), purchases=("purchased", "sum"))
        .reset_index()
        .sort_values("month_order")
    )
    eff_grade["close_rate"] = (eff_grade["purchases"] / eff_grade["enrollments"] * 100).round(1)
    eff_grade["grade_label"] = "Gr " + eff_grade["grade_num"].astype(int).astype(str)

    tc3_exam_choice = st.selectbox(
        "Select exam type to explore grade breakdown",
        options=ACT_SAT_PSAT,
        key="tc3_exam_choice",
    )
    eff_grade_sel = eff_grade[eff_grade["exam_bucket"] == tc3_exam_choice]
    if not eff_grade_sel.empty:
        fig_eff_grade = px.line(
            eff_grade_sel, x="month_order", y="close_rate", color="grade_label",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"close_rate": "Close Rate (%)", "month_order": "Exam Month", "grade_label": "Grade"},
            title=f"{tc3_exam_choice} — Close Rate by Exam Month & Grade",
        )
        apply_dark_theme(fig_eff_grade)
        fix_month_axis_all(fig_eff_grade)
        st.plotly_chart(fig_eff_grade, use_container_width=True)

    # By region: close rate per exam month
    if "enrollee_lead_source_region" in eff_df.columns:
        st.markdown("#### Close Rate by Exam Month & Region (per Exam Type)")
        tc3_exam_reg = st.selectbox(
            "Select exam type for region breakdown",
            options=ACT_SAT_PSAT,
            key="tc3_exam_reg",
        )
        eff_reg = (
            eff_df[eff_df["exam_bucket"] == tc3_exam_reg]
            .groupby(["month_order", "enrollee_lead_source_region"])
            .agg(enrollments=("enrollment_id","count"), purchases=("purchased","sum"))
            .reset_index()
            .sort_values("month_order")
        )
        eff_reg["close_rate"] = (eff_reg["purchases"] / eff_reg["enrollments"] * 100).round(1)
        fig_eff_reg = px.line(
            eff_reg, x="month_order", y="close_rate", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"close_rate": "Close Rate (%)", "month_order": "Exam Month", "enrollee_lead_source_region": "Region"},
            title=f"{tc3_exam_reg} — Close Rate by Exam Month & Region",
        )
        apply_dark_theme(fig_eff_reg)
        fix_month_axis_all(fig_eff_reg)
        st.plotly_chart(fig_eff_reg, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 · EXAM VOLUME × CONSULTS & CLOSES
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 4 · Exam Volume vs. Consults & Closes at Different Times of Year")
    st.caption("Do months where we run more exams produce proportionally more consults and closes? Are there diminishing returns at high volumes?")

    vol_df = tdf_f[tdf_f["exam_bucket"].isin(ACT_SAT_PSAT)].copy()

    vol_agg = (
        vol_df.groupby(["exam_ym_dt", "exam_bucket"])
        .agg(
            enrollments=("enrollment_id", "count"),
            unique_sessions=("session_id", "nunique"),
            purchases=("purchased", "sum"),
            consults=("consulted", "sum"),
        )
        .reset_index()
        .sort_values("exam_ym_dt")
    )
    vol_agg["close_rate"]   = (vol_agg["purchases"] / vol_agg["enrollments"] * 100).round(1)
    vol_agg["consult_rate"] = (vol_agg["consults"]  / vol_agg["enrollments"] * 100).round(1)

    # ── 4a: Volume (sessions) + close rate dual-axis by exam type
    st.markdown("#### Monthly Exam Volume & Close Rate Over Time")

    tc4_exam = st.selectbox(
        "Select exam type",
        options=ACT_SAT_PSAT,
        key="tc4_exam",
    )
    vol_sel = vol_agg[vol_agg["exam_bucket"] == tc4_exam].copy()

    if not vol_sel.empty:
        fig_vol_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vol_dual.add_trace(
            go.Bar(
                x=vol_sel["exam_ym_dt"], y=vol_sel["enrollments"],
                name="Enrollments", marker_color=PLOT_COLORS[0], opacity=0.7,
            ),
            secondary_y=False,
        )
        fig_vol_dual.add_trace(
            go.Scatter(
                x=vol_sel["exam_ym_dt"], y=vol_sel["close_rate"],
                name="Close Rate (%)", mode="lines+markers",
                line=dict(color=PLOT_COLORS[1], width=2),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )
        fig_vol_dual.add_trace(
            go.Scatter(
                x=vol_sel["exam_ym_dt"], y=vol_sel["consult_rate"],
                name="Consult Rate (%)", mode="lines+markers",
                line=dict(color=PLOT_COLORS[2], width=2, dash="dot"),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )
        fig_vol_dual.update_layout(
            paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
            font=dict(family="DM Sans, sans-serif", color=PLOT_FONT, size=12),
            legend=dict(bgcolor="#181c27", bordercolor="#2a2f3e", borderwidth=1),
            margin=dict(t=50, b=30, l=10, r=10),
            title=f"{tc4_exam} — Monthly Enrollments vs. Close & Consult Rates",
            xaxis=dict(tickformat="%b %Y", gridcolor=GRID_COLOR),
        )
        fig_vol_dual.update_yaxes(title_text="Enrollments", secondary_y=False, gridcolor=GRID_COLOR, tickfont_color=PLOT_FONT)
        fig_vol_dual.update_yaxes(title_text="Rate (%)", secondary_y=True, gridcolor=GRID_COLOR, tickfont_color=PLOT_FONT)
        st.plotly_chart(fig_vol_dual, use_container_width=True)

    # ── 4b: All three exam types side-by-side — monthly enrollments stacked
    st.markdown("#### All Exam Types: Monthly Enrollment Volume (Stacked)")
    fig_vol_stack = px.bar(
        vol_agg, x="exam_ym_dt", y="enrollments", color="exam_bucket",
        barmode="stack", color_discrete_sequence=PLOT_COLORS,
        labels={"enrollments": "Enrollments", "exam_ym_dt": "Month", "exam_bucket": "Exam Type"},
        title="Total Monthly Enrollments by Exam Type (Jan 2024 – Feb 2026)",
    )
    apply_dark_theme(fig_vol_stack)
    fig_vol_stack.update_layout(xaxis_tickformat="%b %Y")
    st.plotly_chart(fig_vol_stack, use_container_width=True)

    # ── 4c: Scatter — enrollments vs. close rate per month (bubble = consults)
    st.markdown("#### Volume vs. Effectiveness: Is Higher Volume Correlated With Better Close Rates?")
    st.caption("Each bubble = one exam type in one month. Bubble size = consult count. Look for a positive or negative slope.")
    fig_vol_scatter = px.scatter(
        vol_agg, x="enrollments", y="close_rate",
        size="consults", color="exam_bucket",
        hover_data=["exam_ym_dt"],
        color_discrete_sequence=PLOT_COLORS,
        size_max=40,
        labels={"enrollments": "Enrollments That Month", "close_rate": "Close Rate (%)", "exam_bucket": "Exam Type"},
        title="Volume vs. Close Rate (bubble = consult count)",
    )
    apply_dark_theme(fig_vol_scatter)
    st.plotly_chart(fig_vol_scatter, use_container_width=True)

    # ── 4d: Volume vs. close rate by month-of-year (seasonality view)
    st.markdown("#### Seasonality: Average Volume, Consult Rate & Close Rate by Calendar Month")
    st.caption("Averaged across all years in the dataset. Shows the natural seasonal rhythm for each exam type.")
    seasonal = (
        vol_df.groupby(["exam_month_num", "exam_bucket"])
        .agg(
            avg_enrollments=("enrollment_id", "count"),
            total_purchases=("purchased", "sum"),
            total_consults=("consulted", "sum"),
        )
        .reset_index()
        .sort_values("exam_month_num")
    )
    seasonal["close_rate"]   = (seasonal["total_purchases"] / seasonal["avg_enrollments"] * 100).round(1)
    seasonal["consult_rate"] = (seasonal["total_consults"]  / seasonal["avg_enrollments"] * 100).round(1)

    tc4c1, tc4c2 = st.columns(2)
    with tc4c1:
        fig_seas_vol = px.bar(
            seasonal, x="exam_month_num", y="avg_enrollments", color="exam_bucket",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={"avg_enrollments": "Total Enrollments", "exam_month_num": "Month", "exam_bucket": "Exam Type"},
            title="Total Enrollments by Month & Exam Type",
        )
        apply_dark_theme(fig_seas_vol)
        fix_month_axis_all(fig_seas_vol)
        st.plotly_chart(fig_seas_vol, use_container_width=True)

    with tc4c2:
        fig_seas_cr = px.line(
            seasonal, x="exam_month_num", y="close_rate", color="exam_bucket",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"close_rate": "Close Rate (%)", "exam_month_num": "Month", "exam_bucket": "Exam Type"},
            title="Avg Close Rate by Month & Exam Type",
        )
        apply_dark_theme(fig_seas_cr)
        fix_month_axis_all(fig_seas_cr)
        st.plotly_chart(fig_seas_cr, use_container_width=True)

    # ── 4e: Grade and region breakdown of volume→close relationship
    st.markdown("#### Volume → Close Rate: Grade & Region Breakdown")
    tc4c3, tc4c4 = st.columns(2)

    with tc4c3:
        vol_grade = (
            vol_df[vol_df["grade_num"].notna()]
            .groupby(["exam_month_num", "grade_num", "exam_bucket"])
            .agg(enrollments=("enrollment_id","count"), purchases=("purchased","sum"))
            .reset_index()
            .sort_values("exam_month_num")
        )
        vol_grade["close_rate"] = (vol_grade["purchases"] / vol_grade["enrollments"] * 100).round(1)
        vol_grade["grade_label"] = "Gr " + vol_grade["grade_num"].astype(int).astype(str)

        tc4_exam_g = st.selectbox("Exam type (grade view)", options=ACT_SAT_PSAT, key="tc4g")
        vol_grade_sel = vol_grade[vol_grade["exam_bucket"] == tc4_exam_g]
        if not vol_grade_sel.empty:
            fig_vol_grade = px.line(
                vol_grade_sel, x="exam_month_num", y="close_rate", color="grade_label",
                markers=True, color_discrete_sequence=PLOT_COLORS,
                labels={"close_rate": "Close Rate (%)", "exam_month_num": "Month", "grade_label": "Grade"},
                title=f"{tc4_exam_g} — Close Rate by Month & Grade",
            )
            apply_dark_theme(fig_vol_grade)
            fix_month_axis_all(fig_vol_grade)
            st.plotly_chart(fig_vol_grade, use_container_width=True)

    with tc4c4:
        if "enrollee_lead_source_region" in vol_df.columns:
            vol_reg = (
                vol_df.groupby(["exam_month_num", "enrollee_lead_source_region", "exam_bucket"])
                .agg(enrollments=("enrollment_id","count"), purchases=("purchased","sum"))
                .reset_index()
                .sort_values("exam_month_num")
            )
            vol_reg["close_rate"] = (vol_reg["purchases"] / vol_reg["enrollments"] * 100).round(1)

            tc4_exam_r = st.selectbox("Exam type (region view)", options=ACT_SAT_PSAT, key="tc4r")
            vol_reg_sel = vol_reg[vol_reg["exam_bucket"] == tc4_exam_r]
            if not vol_reg_sel.empty:
                fig_vol_reg = px.line(
                    vol_reg_sel, x="exam_month_num", y="close_rate", color="enrollee_lead_source_region",
                    markers=True, color_discrete_sequence=PLOT_COLORS,
                    labels={"close_rate": "Close Rate (%)", "exam_month_num": "Month", "enrollee_lead_source_region": "Region"},
                    title=f"{tc4_exam_r} — Close Rate by Month & Region",
                )
                apply_dark_theme(fig_vol_reg)
                fix_month_axis_all(fig_vol_reg)
                st.plotly_chart(fig_vol_reg, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 · SAT AUG–DEC REGIONAL DIVERGENCE INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────────
with tab9:
    st.markdown("## 🗺️ SAT Aug–Dec Regional Divergence — Why Do Regions Split?")
    st.markdown("""
    <div class="warn-box">
    The close rate by month × region chart shows <b>significant regional divergence for SAT during August–December</b>.
    This tab systematically investigates every plausible driver: who's enrolling, when they're being reached,
    how the funnel behaves, and what the event mix looks like in that window — all cut by region.
    </div>
    """, unsafe_allow_html=True)

    # ── Build the scoped dataset ──────────────────────────────────────────────
    SAT_AUGDEC_MONTHS = [8, 9, 10, 11, 12]

    sat_ad = tdf_f[
        (tdf_f["exam_bucket"] == "SAT") &
        (tdf_f["exam_month_num"].isin(SAT_AUGDEC_MONTHS))
    ].copy()

    # Allow user to compare against the rest-of-year as a benchmark
    sat_other = tdf_f[
        (tdf_f["exam_bucket"] == "SAT") &
        (~tdf_f["exam_month_num"].isin(SAT_AUGDEC_MONTHS))
    ].copy()

    if sat_ad.empty:
        st.warning("No SAT data found for Aug–Dec after current filters.")
        st.stop()

    # Region selector for this tab
    sat_regions = sorted(sat_ad["enrollee_lead_source_region"].dropna().unique().tolist())
    sel_sat_regions = st.multiselect(
        "Focus regions (default = all)",
        options=sat_regions,
        default=sat_regions,
        key="sat_reg_sel",
    )
    sat_ad = sat_ad[sat_ad["enrollee_lead_source_region"].isin(sel_sat_regions)]

    # Use month number as x-axis throughout this tab — guaranteed numeric sort.
    # A helper applies named ticks after the fact so charts read "Aug Sep Oct…"
    MONTH_NAMES = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    MONTH_TICKVALS = [8, 9, 10, 11, 12]
    MONTH_TICKTEXT = ["Aug", "Sep", "Oct", "Nov", "Dec"]

    def fix_month_axis(fig):
        """Replace numeric month ticks with month-name labels on x-axis."""
        fig.update_xaxes(
            tickmode="array",
            tickvals=MONTH_TICKVALS,
            ticktext=MONTH_TICKTEXT,
        )
        return fig

    # Group using exam_month_num (integer) — no string labels needed until display
    # ── SECTION A: Confirm & Frame the Divergence ─────────────────────────────
    st.markdown("---")
    st.markdown("### A · Confirm the Divergence")

    reg_month_agg = (
        sat_ad.groupby(["enrollee_lead_source_region", "exam_month_num"])
        .agg(enrollments=("enrollment_id","count"), purchases=("purchased","sum"),
             consults=("consulted","sum"))
        .reset_index()
        .sort_values("exam_month_num")
    )
    reg_month_agg["close_rate"]   = (reg_month_agg["purchases"] / reg_month_agg["enrollments"] * 100).round(1)
    reg_month_agg["consult_rate"] = (reg_month_agg["consults"]  / reg_month_agg["enrollments"] * 100).round(1)

    s9a1, s9a2 = st.columns(2)
    with s9a1:
        fig_confirm_cr = px.line(
            reg_month_agg, x="exam_month_num", y="close_rate", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"close_rate": "Close Rate (%)", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="SAT Close Rate by Region: Aug–Dec (all years)",
        )
        apply_dark_theme(fig_confirm_cr)
        fix_month_axis(fig_confirm_cr)
        st.plotly_chart(fig_confirm_cr, use_container_width=True)

    with s9a2:
        fig_confirm_con = px.line(
            reg_month_agg, x="exam_month_num", y="consult_rate", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"consult_rate": "Consult Rate (%)", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="SAT Consult Rate by Region: Aug–Dec (all years)",
        )
        apply_dark_theme(fig_confirm_con)
        fix_month_axis(fig_confirm_con)
        st.plotly_chart(fig_confirm_con, use_container_width=True)

    st.markdown("##### Is divergence partly a low-volume / reliability issue?")
    st.caption("Regions with very few enrollments in a given month will have noisy close rates. Check volume before drawing conclusions.")
    fig_vol_conf = px.bar(
        reg_month_agg, x="exam_month_num", y="enrollments", color="enrollee_lead_source_region",
        barmode="group", color_discrete_sequence=PLOT_COLORS,
        labels={"enrollments": "Enrollments", "exam_month_num": "Month",
                "enrollee_lead_source_region": "Region"},
        title="SAT Enrollment Volume by Region & Month: Aug–Dec",
    )
    apply_dark_theme(fig_vol_conf)
    fix_month_axis(fig_vol_conf)
    st.plotly_chart(fig_vol_conf, use_container_width=True)

    # ── SECTION B: Grade Mix Differences ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### B · Is It a Grade Mix Story?")
    st.caption("Regions where older (grade 11–12) students dominate Aug–Dec SAT enrollment will tend to close faster and at higher rates.")

    if "grade_num" in sat_ad.columns:
        avg_grade_reg = (
            sat_ad[sat_ad["grade_num"].notna()]
            .groupby(["enrollee_lead_source_region", "exam_month_num"])["grade_num"]
            .mean().round(2).reset_index()
            .sort_values("exam_month_num")
        )
        fig_avg_grade = px.line(
            avg_grade_reg, x="exam_month_num", y="grade_num", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"grade_num": "Avg Grade", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="Average Grade of SAT Enrollee by Region & Month",
        )
        apply_dark_theme(fig_avg_grade)
        fix_month_axis(fig_avg_grade)
        st.plotly_chart(fig_avg_grade, use_container_width=True)

        # Grade 11+12 share per region per month
        grade_reg = (
            sat_ad[sat_ad["grade_num"].notna()]
            .groupby(["enrollee_lead_source_region", "exam_month_num", "grade_num"])
            .size().reset_index(name="count")
        )
        grade_reg["grade_label"] = "Gr " + grade_reg["grade_num"].astype(int).astype(str)
        grade_reg["is_upper"] = grade_reg["grade_label"].isin(["Gr 11", "Gr 12"])
        upper_share = (
            grade_reg.groupby(["enrollee_lead_source_region", "exam_month_num"])
            .apply(lambda g: g.loc[g["is_upper"], "count"].sum() / g["count"].sum() * 100)
            .reset_index(name="upper_pct")
            .sort_values("exam_month_num")
        )
        fig_upper = px.bar(
            upper_share, x="exam_month_num", y="upper_pct", color="enrollee_lead_source_region",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={"upper_pct": "% Grade 11–12", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="Share of Grade 11–12 Enrollees by Region & Month (SAT Aug–Dec)",
        )
        apply_dark_theme(fig_upper)
        fix_month_axis(fig_upper)
        st.plotly_chart(fig_upper, use_container_width=True)

    # ── SECTION C: Promotion Type Mix ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### C · Is It a Promotion Mix Story?")
    st.caption("Unpromoted enrollments have no advisor touchpoint. If certain regions run more unpromoted SAT events in Aug–Dec, that would depress consult and close rates.")

    if "promotion_type" in sat_ad.columns:
        promo_reg = (
            sat_ad.groupby(["enrollee_lead_source_region", "exam_month_num", "promotion_type"])
            .size().reset_index(name="count")
            .sort_values("exam_month_num")
        )
        promo_reg["pct"] = promo_reg["count"] / promo_reg.groupby(
            ["enrollee_lead_source_region", "exam_month_num"])["count"].transform("sum") * 100
        promo_reg["is_unpromoted"] = promo_reg["promotion_type"].astype(str).str.contains("Unpromoted", na=False)

        unprom_share = (
            promo_reg.groupby(["enrollee_lead_source_region", "exam_month_num"])
            .apply(lambda g: g.loc[g["is_unpromoted"], "count"].sum() / g["count"].sum() * 100)
            .reset_index(name="unprom_pct")
            .sort_values("exam_month_num")
        )
        fig_unprom = px.line(
            unprom_share, x="exam_month_num", y="unprom_pct", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"unprom_pct": "% Unpromoted", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="Unpromoted SAT Enrollment Share by Region & Month",
        )
        apply_dark_theme(fig_unprom)
        fix_month_axis(fig_unprom)
        st.plotly_chart(fig_unprom, use_container_width=True)

        # Faceted stacked bar — month number on x, named ticks applied per axis
        fig_promo_facet = px.bar(
            promo_reg, x="exam_month_num", y="pct", color="promotion_type",
            facet_col="enrollee_lead_source_region", facet_col_wrap=3,
            barmode="stack", color_discrete_sequence=PLOT_COLORS,
            labels={"pct": "% of Enrollments", "exam_month_num": "Month"},
            title="Promotion Type Mix: SAT Aug–Dec by Region",
            height=max(400, len(sel_sat_regions) // 3 * 280 + 280),
        )
        apply_dark_theme(fig_promo_facet)
        fig_promo_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_promo_facet.for_each_xaxis(lambda ax: ax.update(
            tickmode="array", tickvals=MONTH_TICKVALS, ticktext=MONTH_TICKTEXT
        ))
        st.plotly_chart(fig_promo_facet, use_container_width=True)

    # ── SECTION D: Consult Speed ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### D · Is It a Consult Speed Story?")
    st.caption("Even if consult rates are similar, regions where advisors take longer to follow up will see lower close rates. A rising median days-to-consult = leads going cold.")

    if "days_exam_to_consult" in sat_ad.columns:
        dtc_sat = sat_ad[sat_ad["days_exam_to_consult"].notna() & (sat_ad["days_exam_to_consult"] >= 0)].copy()

        if not dtc_sat.empty:
            dtc_reg_month = (
                dtc_sat.groupby(["enrollee_lead_source_region", "exam_month_num"])["days_exam_to_consult"]
                .median().reset_index()
                .rename(columns={"days_exam_to_consult": "median_dtc"})
                .sort_values("exam_month_num")
            )
            fig_dtc_reg = px.line(
                dtc_reg_month, x="exam_month_num", y="median_dtc", color="enrollee_lead_source_region",
                markers=True, color_discrete_sequence=PLOT_COLORS,
                labels={"median_dtc": "Median Days to Consult", "exam_month_num": "Month",
                        "enrollee_lead_source_region": "Region"},
                title="Median Days Exam → Consult by Region & Month (SAT Aug–Dec)",
            )
            apply_dark_theme(fig_dtc_reg)
            fix_month_axis(fig_dtc_reg)
            st.plotly_chart(fig_dtc_reg, use_container_width=True)

            dtc_sat["month_label"] = dtc_sat["exam_month_num"].map(MONTH_NAMES)
            fig_dtc_box = px.box(
                dtc_sat.sort_values("exam_month_num"),
                x="enrollee_lead_source_region", y="days_exam_to_consult",
                color="exam_month_num",
                color_discrete_sequence=PLOT_COLORS,
                labels={"days_exam_to_consult": "Days to Consult",
                        "enrollee_lead_source_region": "", "exam_month_num": "Month"},
                title="Days Exam → Consult Distribution by Region & Month",
            )
            apply_dark_theme(fig_dtc_box)
            # Rename legend entries to month names
            for trace in fig_dtc_box.data:
                try:
                    trace.name = MONTH_NAMES.get(int(float(trace.name)), trace.name)
                except Exception:
                    pass
            fig_dtc_box.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_dtc_box, use_container_width=True)

    # ── SECTION E: Attendance ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### E · Is It an Attendance Story?")
    st.caption("Regions with lower show-up rates in Aug–Dec never experience the product — reducing the chance they consult or buy.")

    if "attended" in sat_ad.columns:
        att_reg = (
            sat_ad.groupby(["enrollee_lead_source_region", "exam_month_num"])
            .agg(total=("attended","count"), attended=("attended","sum"))
            .reset_index()
            .sort_values("exam_month_num")
        )
        att_reg["att_rate"] = (att_reg["attended"] / att_reg["total"] * 100).round(1)

        fig_att_reg = px.line(
            att_reg, x="exam_month_num", y="att_rate", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"att_rate": "Attendance Rate (%)", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="SAT Attendance Rate by Region & Month: Aug–Dec",
        )
        apply_dark_theme(fig_att_reg)
        fix_month_axis(fig_att_reg)
        st.plotly_chart(fig_att_reg, use_container_width=True)

    # ── SECTION F: Top 100 School Mix ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### F · Is It a Top 100 School Mix Story?")
    st.caption("Regions where Top 100 families dominate tend to close at higher rates. A shift in that mix Aug–Dec by region would explain the divergence.")

    if "ly_top_100" in sat_ad.columns:
        top_reg = (
            sat_ad.groupby(["enrollee_lead_source_region", "exam_month_num"])
            .agg(total=("enrollment_id","count"), top100=("ly_top_100","sum"))
            .reset_index()
            .sort_values("exam_month_num")
        )
        top_reg["top100_pct"] = (top_reg["top100"] / top_reg["total"] * 100).round(1)

        fig_top_reg = px.line(
            top_reg, x="exam_month_num", y="top100_pct", color="enrollee_lead_source_region",
            markers=True, color_discrete_sequence=PLOT_COLORS,
            labels={"top100_pct": "% Top 100 School", "exam_month_num": "Month",
                    "enrollee_lead_source_region": "Region"},
            title="Top 100 School Share by Region & Month (SAT Aug–Dec)",
        )
        apply_dark_theme(fig_top_reg)
        fix_month_axis(fig_top_reg)
        st.plotly_chart(fig_top_reg, use_container_width=True)

    # ── SECTION G: Benchmark — Aug–Dec vs. rest of year ──────────────────────
    st.markdown("---")
    st.markdown("### G · Aug–Dec vs. Rest of Year — Is This Window Structurally Different?")
    st.caption("Comparing each region's SAT performance in Aug–Dec vs. all other months. Regions that are structurally weaker in this window (vs. their own baseline) are worth a closer look.")

    def region_summary(df_in, label):
        if df_in.empty:
            return pd.DataFrame()
        return (
            df_in.groupby("enrollee_lead_source_region")
            .agg(
                enrollments=("enrollment_id","count"),
                purchases=("purchased","sum"),
                consults=("consulted","sum"),
                attended=("attended","sum"),
                top100=("ly_top_100","sum"),
            )
            .reset_index()
            .assign(
                window=label,
                close_rate=lambda x: (x["purchases"] / x["enrollments"] * 100).round(1),
                consult_rate=lambda x: (x["consults"] / x["enrollments"] * 100).round(1),
                att_rate=lambda x: (x["attended"] / x["enrollments"] * 100).round(1),
                top100_pct=lambda x: (x["top100"] / x["enrollments"] * 100).round(1),
            )
        )

    bench_augdec = region_summary(
        sat_ad[sat_ad["enrollee_lead_source_region"].isin(sel_sat_regions)], "Aug–Dec"
    )
    bench_other = region_summary(
        sat_other[sat_other["enrollee_lead_source_region"].isin(sel_sat_regions)], "Rest of Year"
    )
    bench = pd.concat([bench_augdec, bench_other])

    bench_metrics = [
        ("close_rate",   "Close Rate (%)",   "Close Rate: Aug–Dec vs. Rest of Year"),
        ("consult_rate", "Consult Rate (%)",  "Consult Rate: Aug–Dec vs. Rest of Year"),
        ("att_rate",     "Attendance Rate (%)","Attendance: Aug–Dec vs. Rest of Year"),
        ("top100_pct",   "Top 100 Share (%)", "Top 100 Share: Aug–Dec vs. Rest of Year"),
    ]
    for metric, ylabel, title in bench_metrics:
        fig_bench = px.bar(
            bench, x="enrollee_lead_source_region", y=metric, color="window",
            barmode="group", color_discrete_sequence=PLOT_COLORS,
            labels={metric: ylabel, "enrollee_lead_source_region": "", "window": "Window"},
            title=title,
        )
        apply_dark_theme(fig_bench)
        fig_bench.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_bench, use_container_width=True)

    # ── SECTION H: Summary Scorecard ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### H · Region Scorecard: SAT Aug–Dec")
    st.caption("All key metrics in one table, one row per region. Sort by close rate to quickly spot which regions are outliers.")

    scorecard_rows = []
    for reg, grp in sat_ad.groupby("enrollee_lead_source_region"):
        n = len(grp)
        cr   = grp["purchased"].sum() / n * 100 if n else None
        conr = grp["consulted"].sum()  / n * 100 if n else None
        attr = grp["attended"].sum()   / n * 100 if "attended" in grp.columns and n else None
        t100 = (grp["ly_top_100"] == 1).sum() / n * 100 if "ly_top_100" in grp.columns and n else None
        dtcm = grp["days_exam_to_consult"].median() if "days_exam_to_consult" in grp.columns else None
        avg_g = grp["grade_num"].mean() if "grade_num" in grp.columns else None
        unprom = grp["promotion_type"].astype(str).str.contains("Unpromoted", na=False).sum() / n * 100 if "promotion_type" in grp.columns and n else None
        scorecard_rows.append({
            "Region": reg,
            "Enrollments": n,
            "Close Rate %": round(cr, 1) if cr is not None else None,
            "Consult Rate %": round(conr, 1) if conr is not None else None,
            "Attendance Rate %": round(attr, 1) if attr is not None else None,
            "Top 100 Share %": round(t100, 1) if t100 is not None else None,
            "Median Days to Consult": round(dtcm, 1) if dtcm is not None and not pd.isna(dtcm) else None,
            "Avg Grade": round(avg_g, 2) if avg_g is not None and not pd.isna(avg_g) else None,
            "Unpromoted Share %": round(unprom, 1) if unprom is not None else None,
        })

    scorecard_df = pd.DataFrame(scorecard_rows).set_index("Region")
    if not scorecard_df.empty:
        # Highlight low close rates
        def highlight_low(val):
            try:
                if float(val) < scorecard_df["Close Rate %"].median():
                    return "color: #e05b5b"
            except Exception:
                pass
            return ""
        st.dataframe(
            scorecard_df.sort_values("Close Rate %", ascending=True),
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Special Events Enrollment Intelligence Dashboard · Built with Streamlit & Plotly · Data: data.csv")