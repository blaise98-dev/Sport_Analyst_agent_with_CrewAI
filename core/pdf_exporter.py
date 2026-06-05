# core/pdf_exporter.py
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from .report_generator import generate_player_recommendation


def generate_pdf_report(text_output: str, charts: dict, df, llm=None) -> bytes:
    """
    Generate a PDF report including global text, tables, charts,
    and per-player profile pages with LLM recommendations.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("🏀 Basketball Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Add global text
    for line in text_output.splitlines():
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 12))

    # KPI Table
    if {"player_name", "avg_speed", "max_speed"}.issubset(df.columns):
        table_data = [["Player", "Avg Speed", "Max Speed", "Total Distance"]]
        for _, row in df.head(10).iterrows():
            table_data.append([
                row["player_name"],
                f"{row['avg_speed']:.2f}" if "avg_speed" in row else "",
                f"{row['max_speed']:.2f}" if "max_speed" in row else "",
                f"{row['total_distance']:.1f}" if "total_distance" in row else "",
            ])
        story.append(Paragraph("📊 Sample KPI Table", styles["Heading2"]))
        story.append(Table(table_data))
        story.append(Spacer(1, 12))

    # Charts
    for name, fig in charts.items():
        try:
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format="png", width=800, height=600, scale=2)
            img_buffer.seek(0)
            story.append(Paragraph(
                f"Chart: {fig.layout.title.text if fig.layout.title else name}",
                styles["Heading2"]
            ))
            story.append(Image(img_buffer, width=400, height=300))
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Failed to render chart {name}: {e}", styles["Normal"]))

    # -------------------------------
    # Per-Player Profile Pages
    # -------------------------------
    story.append(PageBreak())
    story.append(Paragraph("👤 Player Profiles", styles["Title"]))
    story.append(Spacer(1, 12))

    for _, row in df.iterrows():
        pdata = row.to_dict()

        # Player header
        story.append(Paragraph(f"🏀 {pdata.get('player_name', 'Unknown')}", styles["Heading1"]))
        story.append(Spacer(1, 8))

        # Stats table
        stats_table = [["Metric", "Value"]]
        for metric in ["avg_speed", "max_speed", "total_distance", "avg_acceleration", "speed_variability"]:
            if metric in pdata and not pd.isna(pdata[metric]):
                stats_table.append([metric.replace("_", " ").title(), f"{pdata[metric]:.2f}"])
        story.append(Table(stats_table))
        story.append(Spacer(1, 8))

        # LLM-generated recommendation
        if llm is not None:
            try:
                rec_text = generate_player_recommendation(pdata, llm)
                story.append(Paragraph("📌 Recommendation", styles["Heading2"]))
                for line in rec_text.splitlines():
                    if line.strip():
                        story.append(Paragraph(line, styles["Normal"]))
                        story.append(Spacer(1, 4))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Failed to generate recommendation: {e}", styles["Normal"]))
        else:
            story.append(Paragraph("⚠️ No LLM connected for personalized recommendations.", styles["Normal"]))

        story.append(PageBreak())

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
