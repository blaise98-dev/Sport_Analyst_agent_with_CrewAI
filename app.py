# app.py
import os
# Disable CrewAI telemetry before any crewai import to avoid connection timeout errors
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")

import base64
import traceback
import warnings
from pathlib import Path
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")

from core.yaml_utils import load_yaml_config
from core.llm_manager import LLMManager
from core.crew_manager import CrewManager
from core.charts import create_player_performance_charts, display_charts
from core.report_generator import (
    build_full_data_driven_text,
    generate_llm_recommendations,
    generate_player_recommendation,
)
from core.pdf_exporter import generate_pdf_report


# -------------------------------
# Page & Paths
# -------------------------------
st.set_page_config(page_title="🏀 Basketball Performance – Analyst", layout="wide")
st.title("📊 Basketball Performance – Analyst")

APP_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = APP_DIR / "config"


# -------------------------------
# Load YAML (Agents/Tasks)
# -------------------------------
try:
    agents_dict = load_yaml_config(CONFIG_DIR / "agents.yaml", must_exist=True)
    tasks_dict = load_yaml_config(CONFIG_DIR / "tasks.yaml", must_exist=True)
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()


# -------------------------------
# Data Upload
# -------------------------------
st.subheader("📥 Upload Player Data")
uploaded_file = st.file_uploader(
    "Upload a CSV with columns like: player_name, avg_speed, max_speed, total_distance, ...",
    type=["csv"],
)
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.caption(f"Loaded {len(df)} rows.")
        st.dataframe(df.head(12), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()


# -------------------------------
# Backend (Ollama default)
# -------------------------------
st.sidebar.header("Model Backend")
backend = st.sidebar.selectbox(
    "Backend", ["Ollama", "OpenAI", "Google Gemini"], index=0
)
api_key = "ollama" if backend == "Ollama" else os.getenv("OPENAI_API_KEY", "") if backend == "OpenAI" else os.getenv("GEMINI_API_KEY", "")
model_name = "ollama/llama3.1:8b" if backend == "Ollama" else os.getenv("OPENAI_MODEL_NAME", "gpt-4") if backend == "OpenAI" else os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

try:
    llm = LLMManager.get_llm(backend, api_key, model_name)
except Exception as e:
    st.error(f"LLM initialization failed: {e}")
    st.stop()


# -------------------------------
# Run
# -------------------------------
go = st.button("🚀 Run Analysis", disabled=df is None)

if go:
    if df is None:
        st.warning("Please upload a CSV first.")
        st.stop()

    full_text_sections = []  # will feed into PDF

    try:
        # 1) Multi-agent workflow from YAML (CrewAI)
        crew = CrewManager.build_crew(agents_dict, tasks_dict, llm, df)
        with st.spinner("Running multi-agent workflow..."):
            result = crew.kickoff()

        final_multi_agent = getattr(result, "raw", str(result)) or ""
        st.subheader("📄 Multi-Agent Report")
        st.markdown(final_multi_agent)
        full_text_sections.append("# Multi-Agent Report")
        full_text_sections.append(final_multi_agent)

        # (Optional) task-by-task display if available
        if hasattr(result, "tasks_output") and result.tasks_output:
            st.subheader("🧩 Task-by-Task Outputs")
            for i, t in enumerate(result.tasks_output, start=1):
                ttext = getattr(t, "raw", str(t)) or ""
                if ttext.strip():
                    st.markdown(f"### Task {i}")
                    st.markdown(ttext)
                    full_text_sections.append(f"## Task {i}")
                    full_text_sections.append(ttext)

        # 2) Data-driven KPIs & concrete per-player summary
        st.subheader("📌 KPIs & Per-Player Highlights (Data-Driven)")
        kpi_md, summary_md = build_full_data_driven_text(df)
        st.markdown("#### KPI Table")
        st.markdown(kpi_md)
        st.markdown(summary_md)

        full_text_sections.append("# KPI Table")
        full_text_sections.append(kpi_md)
        full_text_sections.append("# Per-Player Highlights & Team Insights")
        full_text_sections.append(summary_md)

        # 3) LLM recommendations grounded in stats
        with st.spinner("Generating name-based recommendations..."):
            recs_md = generate_llm_recommendations(df, llm)

        st.subheader("🧭 Team Recommendations")
        st.markdown(recs_md)
        full_text_sections.append("# Recommendations")
        full_text_sections.append(recs_md)

        # 4) Visual analytics
        st.subheader("📊 Visual Analytics")
        charts = create_player_performance_charts(df)
        display_charts(charts, df)

        # 5) Per-player profiles with search/filter
        st.subheader("👤 Player Profiles – Individual Analysis")

        # Search and filter bar
        player_names = df["player_name"].dropna().unique().tolist()
        search_query = st.text_input("🔍 Search Player by Name")
        selected_player = st.selectbox("Quick Select Player", ["All"] + player_names)

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["player_name"].str.contains(search_query, case=False, na=False)
            ]
        if selected_player != "All":
            filtered_df = filtered_df[filtered_df["player_name"] == selected_player]

        if filtered_df.empty:
            st.warning("No players match your search/filter.")
        else:
            for _, row in filtered_df.iterrows():
                pdata = row.to_dict()
                with st.expander(f"🏀 {pdata.get('player_name', 'Unknown')}", expanded=False):
                    # Show stats
                    st.write("**Stats**")
                    stats_lines = []
                    for metric in [
                        "avg_speed",
                        "max_speed",
                        "total_distance",
                        "avg_acceleration",
                        "speed_variability",
                    ]:
                        if metric in pdata and not pd.isna(pdata[metric]):
                            stats_lines.append(
                                f"- {metric.replace('_', ' ').title()}: **{pdata[metric]:.2f}**"
                            )
                    st.markdown("\n".join(stats_lines))

                    # Generate and show recommendation
                    with st.spinner(f"Generating recommendation for {pdata.get('player_name')}..."):
                        try:
                            rec_text = generate_player_recommendation(pdata, llm)
                            st.write("**📌 Recommendation**")
                            st.markdown(rec_text)
                        except Exception as e:
                            st.warning(f"Could not generate recommendation: {e}")

        # 6) Build PDF report with everything
        with st.spinner("Building PDF report..."):
            combined_text = "\n\n".join(full_text_sections)
            pdf_bytes = generate_pdf_report(combined_text, charts, df, llm=llm)

        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" download="Basketball_Analysis_Report.pdf">⬇️ Download PDF Report</a>',
            unsafe_allow_html=True,
        )

        st.success("Done.")

    except Exception as e:
        st.error("❌ Analysis failed.")
        st.code(traceback.format_exc())
