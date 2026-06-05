# core/report_generator.py
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from crewai import Agent, Task, Crew


# -------------------------------
# Data-first metrics & summaries
# -------------------------------
REQUIRED_COLS = ["player_name", "avg_speed", "max_speed", "total_distance"]

def _safe_cols(df: pd.DataFrame) -> List[str]:
    have = [c for c in REQUIRED_COLS if c in df.columns]
    return have

def compute_team_and_player_metrics(df: pd.DataFrame) -> Dict:
    """Compute concrete, player- and team-level metrics from the dataset."""
    cols = _safe_cols(df)
    if len(cols) < 3:
        raise ValueError(
            f"Dataset is missing required columns. Have: {list(df.columns)} "
            f"Expected at least: {REQUIRED_COLS}"
        )

    res: Dict = {}
    df = df.copy()

    # Basic team statistics
    team = {}
    for metric in [c for c in ["avg_speed", "max_speed", "total_distance", "avg_acceleration"] if c in df.columns]:
        team[f"team_{metric}_mean"] = float(df[metric].mean())
        team[f"team_{metric}_median"] = float(df[metric].median())
        team[f"team_{metric}_std"] = float(df[metric].std(ddof=0)) if len(df) > 1 else 0.0

    # Rankings
    rankings = {}
    if "avg_speed" in df.columns:
        rankings["fastest_avg_speed"] = df.nlargest(3, "avg_speed")[["player_name", "avg_speed"]].to_dict("records")
        rankings["slowest_avg_speed"] = df.nsmallest(3, "avg_speed")[["player_name", "avg_speed"]].to_dict("records")
    if "max_speed" in df.columns:
        rankings["top_max_sprint"] = df.nlargest(3, "max_speed")[["player_name", "max_speed"]].to_dict("records")
    if "total_distance" in df.columns:
        rankings["most_distance"] = df.nlargest(3, "total_distance")[["player_name", "total_distance"]].to_dict("records")

    # Outliers (z-score > 1.0 by default)
    outliers = {}
    def zflag(series: pd.Series, thresh: float = 1.0) -> Tuple[List, List]:
        if series.std(ddof=0) == 0:
            return [], []
        z = (series - series.mean()) / series.std(ddof=0)
        return list(z[z > thresh].index), list(z[z < -thresh].index)

    if "avg_speed" in df.columns:
        hi, lo = zflag(df["avg_speed"])
        outliers["avg_speed_high"] = df.loc[hi, ["player_name", "avg_speed"]].to_dict("records")
        outliers["avg_speed_low"] = df.loc[lo, ["player_name", "avg_speed"]].to_dict("records")
    if "total_distance" in df.columns:
        hi, lo = zflag(df["total_distance"])
        outliers["distance_high"] = df.loc[hi, ["player_name", "total_distance"]].to_dict("records")
        outliers["distance_low"] = df.loc[lo, ["player_name", "total_distance"]].to_dict("records")

    # Per-player records (only the columns we know)
    keep_cols = [c for c in ["player_name", "avg_speed", "max_speed", "total_distance", "avg_acceleration", "speed_variability"] if c in df.columns]
    players = df[keep_cols].to_dict("records")

    res["team_stats"] = team
    res["rankings"] = rankings
    res["outliers"] = outliers
    res["players"] = players
    res["row_count"] = int(len(df))
    return res


def build_kpi_table_md(df: pd.DataFrame) -> str:
    """Return a small, readable KPI table as markdown."""
    cols = [c for c in ["player_name", "avg_speed", "max_speed", "total_distance", "avg_acceleration"] if c in df.columns]
    kpi_df = df[cols].copy()
    # Round for readability
    for c in [c for c in kpi_df.columns if c != "player_name"]:
        kpi_df[c] = kpi_df[c].astype(float).round(2)
    return kpi_df.to_markdown(index=False)


def build_data_driven_summary_md(metrics: Dict) -> str:
    """Turn the computed metrics into concrete, human-readable Markdown."""
    team = metrics["team_stats"]
    rankings = metrics["rankings"]
    outliers = metrics["outliers"]

    lines = []
    lines.append("### 🏀 Executive Summary")
    lines.append(f"- Players analyzed: **{metrics['row_count']}**")
    if "team_avg_speed_mean" in team:
        lines.append(f"- Team average speed: **{team['team_avg_speed_mean']:.2f}**")

    # Top performers
    lines.append("\n### 🥇 Top Performers")
    if "fastest_avg_speed" in rankings:
        s = ", ".join([f"{r['player_name']} ({r['avg_speed']:.2f})" for r in rankings["fastest_avg_speed"]])
        lines.append(f"- Highest average speed: {s}")
    if "top_max_sprint" in rankings:
        s = ", ".join([f"{r['player_name']} ({r['max_speed']:.2f})" for r in rankings["top_max_sprint"]])
        lines.append(f"- Fastest max sprint: {s}")
    if "most_distance" in rankings:
        s = ", ".join([f"{r['player_name']} ({r['total_distance']:.1f})" for r in rankings["most_distance"]])
        lines.append(f"- Most distance covered: {s}")

    # Outliers
    lines.append("\n### 📌 Notable Outliers")
    if outliers.get("avg_speed_high"):
        s = ", ".join([f"{r['player_name']} ({r['avg_speed']:.2f})" for r in outliers["avg_speed_high"]])
        lines.append(f"- Exceptionally fast avg speed: {s}")
    if outliers.get("avg_speed_low"):
        s = ", ".join([f"{r['player_name']} ({r['avg_speed']:.2f})" for r in outliers["avg_speed_low"]])
        lines.append(f"- Below-average avg speed: {s}")
    if outliers.get("distance_high"):
        s = ", ".join([f"{r['player_name']} ({r['total_distance']:.1f})" for r in outliers["distance_high"]])
        lines.append(f"- Exceptional work rate (distance): {s}")
    if outliers.get("distance_low"):
        s = ", ".join([f"{r['player_name']} ({r['total_distance']:.1f})" for r in outliers["distance_low"]])
        lines.append(f"- Low work rate (distance): {s}")

    # Per-player lines
    lines.append("\n### 👤 Per-Player Highlights")
    for r in metrics["players"]:
        bits = [f"**{r['player_name']}**"]
        if "avg_speed" in r: bits.append(f"avg speed: {r['avg_speed']:.2f}")
        if "max_speed" in r: bits.append(f"max speed: {r['max_speed']:.2f}")
        if "total_distance" in r: bits.append(f"distance: {r['total_distance']:.1f}")
        if "avg_acceleration" in r: bits.append(f"acceleration: {r['avg_acceleration']:.2f}")
        if "speed_variability" in r: bits.append(f"variability: {r['speed_variability']:.2f}")
        lines.append("- " + " | ".join(bits))

    return "\n".join(lines)


def build_stats_prompt(metrics: Dict) -> str:
    """Compact, LLM-ready summary with concrete stats, names, and ranks."""
    parts = []

    parts.append("TEAM STATS")
    for k, v in metrics["team_stats"].items():
        parts.append(f"{k}: {v:.4f}")

    parts.append("\nRANKINGS")
    for key, items in metrics["rankings"].items():
        items_s = "; ".join([", ".join([f"{k}={v:.2f}" if isinstance(v, (int,float)) else f"{k}={v}" for k,v in rec.items()]) for rec in items])
        parts.append(f"{key}: {items_s}")

    parts.append("\nPLAYERS")
    for r in metrics["players"]:
        s = ", ".join([f"{k}={r[k]:.2f}" if isinstance(r[k], (int,float)) and k!="player_name" else f"{k}={r[k]}" for k in r])
        parts.append(s)

    return "\n".join(parts)


# -------------------------------
# CrewAI: Always-on Recommendations
# -------------------------------
def generate_llm_recommendations(df: pd.DataFrame, llm) -> str:
    """
    Uses CrewAI + your selected LLM to produce concrete recommendations,
    grounded in actual per-player stats we computed above.
    """
    metrics = compute_team_and_player_metrics(df)
    prompt_stats = build_stats_prompt(metrics)

    analyst = Agent(
        role="Basketball Performance Strategist",
        goal="Provide specific, name-based recommendations for players, coaches, and team management grounded in the provided statistics.",
        backstory="You are a seasoned performance analyst. You speak in clear, direct terms with concrete names and numbers. No fluff.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    description = f"""
You will receive computed player and team statistics from a basketball dataset.
Write a detailed, **data-grounded** report with **concrete names and values** — no generalities.

Required sections:
1) Recommendations for Players (name each player; what to improve, why, and how)
2) Recommendations for Coaches (rotations, matchups, tactics, training blocks, workload)
3) Recommendations for Team Management (roles, contract/rotation considerations, recruitment profiles)

Be concise but comprehensive. Avoid repeating the same advice. Use bullet lists with names.

DATA (computed summary):
```{prompt_stats}```

    """

    task = Task(
        description=description.strip(),
        expected_output="A concise markdown report with the three sections above, using names and numbers from the data.",
        agent=analyst
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=True)
    result = crew.kickoff()
    return getattr(result, "raw", str(result))


# -------------------------------
# Public API for app.py
# -------------------------------
def build_full_data_driven_text(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns (kpi_table_md, data_summary_md)
    - KPI table: compact, readable table
    - Data summary: top performers, outliers, and per-player highlights (concrete)
    """
    metrics = compute_team_and_player_metrics(df)
    kpi_md = build_kpi_table_md(df)
    summary_md = build_data_driven_summary_md(metrics)
    return kpi_md, summary_md

def generate_player_recommendation(player: dict, llm) -> str:
    """
    Generate a personalized recommendation for a single player using CrewAI + LLM.
    `player` is a dict with keys like player_name, avg_speed, max_speed, etc.
    """
    from crewai import Agent, Task, Crew

    # Build a compact stat summary
    stats = ", ".join([f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}"
                       for k, v in player.items() if k != "player_name"])

    analyst = Agent(
        role="Performance Coach",
        goal="Provide player-specific improvement advice based on stats.",
        backstory="You are a basketball coach who reviews player stats and gives clear recommendations.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    description = f"""
Analyze this player’s stats and give **concrete recommendations**.
Be direct, mention the player by name, and keep it in basketball terms.

Player: {player.get("player_name", "Unknown")}
Stats: {stats}
"""

    task = Task(
        description=description.strip(),
        expected_output="1–3 bullet points with recommendations for this player.",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=False)
    result = crew.kickoff()
    return getattr(result, "raw", str(result))
