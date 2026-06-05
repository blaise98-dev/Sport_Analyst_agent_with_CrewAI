# Sport Data Analysts — Basketball Performance Intelligence Platform

A multi-agent AI system built with CrewAI and Streamlit that transforms raw basketball tracking data into comprehensive performance reports, tactical recommendations, and interactive visualizations.

---

## Philosophy

Traditional sports analytics tools present data — this application *interprets* it.

The core idea is that meaningful analysis requires multiple perspectives working in sequence: a data expert who gathers and structures information, a consultant who spots actionable patterns, an analyst who produces executive-ready summaries, and a visualization specialist who communicates findings clearly. Rather than a single model doing everything, four specialized AI agents collaborate the way a real analytics department would — each contributing their role to a final intelligence report.

The platform is designed for basketball staff (coaches, scouts, operations teams) who need fast, grounded insights from player tracking data without requiring data science expertise. Every output — from KPI tables to radar charts to per-player profiles — is built to answer a real question a coaching staff would ask.

---

## What It Does

1. **Upload** a CSV of player tracking metrics (speed, distance, acceleration, position)
2. **Select** your preferred LLM backend (local via Ollama, or cloud via OpenAI / Google Gemini)
3. **Run** the multi-agent crew — five sequential tasks executed by four specialized agents
4. **Review** the results:
   - Full narrative intelligence report
   - KPI tables with team averages, rankings, and outlier detection
   - Team-level and per-player LLM recommendations grounded in statistics
   - Five interactive Plotly charts (bar, scatter, box plots, radar, heatmap on NBA court overlay)
   - Expandable per-player profile cards
   - **Downloadable PDF report** combining all of the above

---

## Agent Roles

| Agent | Role | Responsibility |
|---|---|---|
| **Data Collector** | Basketball Data Collector | Gather player stats, game data, and league metrics |
| **Operations Consultant** | Basketball Operations Consultant | Translate stats into 3–5 actionable coaching recommendations |
| **Intelligence Analyst** | Basketball Intelligence Analyst | Build KPI summaries and the final executive report |
| **Visualization Expert** | Data Visualization Expert | Recommend chart types and design strategy for key metrics |

Tasks run sequentially with dependencies: the analyst and consultant both depend on the collector, the visualization expert depends on both, and the final report depends on all four prior outputs.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally (if using the default local backend)

### 1. Clone and navigate

```bash
git clone <repo-url>
cd MultiAgentCrewAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (or export variables in your shell). Only the keys for backends you intend to use are required.

```bash
# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4          # default

# Google Gemini (optional)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash    # default
```

### 5. Pull an Ollama model (if using local backend)

```bash
ollama pull llama3.1:8b
```

Verify it is available:

```bash
ollama list
```

---

## Running the Application

```bash
cd sport_data_analysts
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Input Data Format

Upload a CSV file with player tracking data. Required columns:

| Column | Type | Description |
|---|---|---|
| `player_name` | string | Player identifier |
| `avg_speed` | float | Average speed (m/s or km/h) |
| `max_speed` | float | Peak speed recorded |
| `total_distance` | float | Total distance covered |

Optional columns that unlock additional analysis:

| Column | Description |
|---|---|
| `avg_acceleration` | Average acceleration — used in scatter plot bubble sizing |
| `speed_variability` | Standard deviation of speed |
| `avg_x` / `avg_y` | Average court position coordinates — used in heatmap |

---

## LLM Backend Options

Select the backend from the sidebar before running analysis:

| Backend | Requirement | Default Model |
|---|---|---|
| **Ollama** (default) | Ollama running on `localhost:11434` | `llama3.1:8b` |
| **OpenAI** | `OPENAI_API_KEY` env var | `gpt-4` |
| **Google Gemini** | `GEMINI_API_KEY` env var | `gemini-2.0-flash` |

---

## Project Structure

```
sport_data_analysts/
├── app.py                  # Streamlit entry point
├── config/
│   ├── agents.yaml         # Agent definitions (role, goal, backstory, tools)
│   └── tasks.yaml          # Task definitions (description, expected output, dependencies)
└── core/
    ├── crew_manager.py     # Builds CrewAI crew dynamically from YAML configs
    ├── llm_manager.py      # LLM factory — selects Ollama, OpenAI, or Gemini
    ├── gemini_llm.py       # Custom wrapper for Google Generative AI SDK
    ├── report_generator.py # Computes KPIs, rankings, outliers; generates LLM recommendations
    ├── charts.py           # Five Plotly visualizations including court heatmap
    ├── pdf_exporter.py     # ReportLab-based PDF with text, tables, charts, player profiles
    └── yaml_utils.py       # YAML config loader with validation
```

---

## Troubleshooting

**`model 'llama3:8b' not found`**
Run `ollama list` to see installed models and ensure `llama3.1:8b` (or your chosen model) is present. Pull it with `ollama pull llama3.1:8b`.

**`LLM initialization failed`**
Check that Ollama is running (`ollama serve`) or that the relevant API key environment variable is set for OpenAI/Gemini backends.

**Charts not rendering in PDF**
Ensure `kaleido` is installed: `pip install kaleido`. It is required by Plotly for static image export.

**Empty agent report**
The multi-agent workflow requires a live LLM connection. Verify network access for cloud backends or that Ollama is responding at `http://localhost:11434`.
