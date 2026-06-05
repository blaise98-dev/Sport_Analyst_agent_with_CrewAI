import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

# -------------------------------
# Main Chart Generator
# -------------------------------
def create_player_performance_charts(df):
    """Generate interactive Plotly charts for player performance analysis."""
    charts = {}

    # 1. Average Speed by Player (Bar Chart)
    fig1 = px.bar(
        df.sort_values("avg_speed", ascending=False),
        x="player_name",
        y="avg_speed",
        color="avg_speed",
        title="🏃 Average Speed by Player",
        labels={"avg_speed": "Average Speed"},
        text_auto=".2f"
    )
    fig1.update_layout(xaxis_tickangle=-45)
    charts["avg_speed"] = fig1

    # 2. Speed vs Distance (Scatter Plot)
    fig2 = px.scatter(
        df,
        x="avg_speed",
        y="total_distance",
        size="max_speed",
        color="avg_acceleration",
        hover_name="player_name",
        title="📈 Speed vs Distance (Bubble size = Max Speed, Color = Acceleration)",
        labels={"avg_speed": "Average Speed", "total_distance": "Total Distance"}
    )
    charts["speed_vs_distance"] = fig2

    # 3. Distribution of Metrics (Box Plots)
    metrics = ["avg_speed", "max_speed", "total_distance", "avg_acceleration"]
    fig3 = make_distribution_boxplot(df, metrics)
    charts["distribution"] = fig3

    # 4. Radar Chart for Top 3 Players
    fig4 = make_radar_chart(df)
    charts["radar"] = fig4

    # 5. Heatmap of Player Positions (with court overlay)
    if "avg_x" in df.columns and "avg_y" in df.columns:
        fig5 = make_position_heatmap(df)
        charts["heatmap"] = fig5

    return charts

# -------------------------------
# Distribution Boxplots
# -------------------------------
def make_distribution_boxplot(df, metrics):
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Box(y=df[metric], name=metric.replace("_", " ").title(), boxmean="sd"))
    fig.update_layout(
        title="📦 Distribution of Performance Metrics",
        yaxis_title="Values"
    )
    return fig

# -------------------------------
# Radar Chart
# -------------------------------
def make_radar_chart(df):
    top_players = df.nlargest(3, "avg_speed")
    metrics = ["avg_speed", "max_speed", "total_distance", "avg_acceleration"]

    fig = go.Figure()
    for _, row in top_players.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # close loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[m.replace("_", " ").title() for m in metrics] + [metrics[0].replace("_", " ").title()],
            fill="toself",
            name=row["player_name"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, df[metrics].max().max()])),
        title="🕸️ Top 3 Players Radar Comparison"
    )
    return fig

# -------------------------------
# Heatmap with Full-Court Overlay
# -------------------------------
import plotly.express as px

def make_position_heatmap(df):
    """Heatmap of player positions with a full NBA court overlay (94x50 ft)."""

    # --- Scale coordinates to fit full court dimensions ---
    df_scaled = df.copy()
    if "avg_x" in df.columns and "avg_y" in df.columns:
        df_scaled["court_x"] = (df["avg_x"] - df["avg_x"].min()) / (df["avg_x"].max() - df["avg_x"].min()) * 94
        df_scaled["court_y"] = (df["avg_y"] - df["avg_y"].min()) / (df["avg_y"].max() - df["avg_y"].min()) * 50
    else:
        raise ValueError("avg_x and avg_y columns are required in dataframe")

    # --- Heatmap ---
    fig = px.density_heatmap(
        df_scaled,
        x="court_x",
        y="court_y",
        nbinsx=30,
        nbinsy=20,
        color_continuous_scale="Plasma",
        title="🗺️ Player Position Heatmap (Full Court Overlay)",
        labels={"court_x": "Court X Position (ft)", "court_y": "Court Y Position (ft)"}
    )
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    # --- Full Court Shapes ---
    court_shapes = [
        # Outer boundary
        dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color="white", width=2)),

        # Left paint
        dict(type="rect", x0=0, y0=17, x1=19, y1=33, line=dict(color="white", width=2)),
        dict(type="circle", x0=19-6, y0=25-6, x1=19+6, y1=25+6, line=dict(color="white", width=2)),  # FT circle

        # Right paint
        dict(type="rect", x0=94-19, y0=17, x1=94, y1=33, line=dict(color="white", width=2)),
        dict(type="circle", x0=94-19-6, y0=25-6, x1=94-19+6, y1=25+6, line=dict(color="white", width=2)),

        # Hoops
        dict(type="circle", x0=-0.75, y0=25-0.75, x1=0.75, y1=25+0.75, line=dict(color="red", width=3)),
        dict(type="circle", x0=94-0.75, y0=25-0.75, x1=94+0.75, y1=25+0.75, line=dict(color="red", width=3)),

        # Backboards
        dict(type="line", x0=0, y0=22, x1=0, y1=28, line=dict(color="white", width=3)),
        dict(type="line", x0=94, y0=22, x1=94, y1=28, line=dict(color="white", width=3)),

        # 3pt arcs
        dict(type="circle", x0=-23.75, y0=25-23.75, x1=23.75, y1=25+23.75, line=dict(color="white", width=2)),
        dict(type="circle", x0=94-23.75, y0=25-23.75, x1=94+23.75, y1=25+23.75, line=dict(color="white", width=2)),

        # Half-court line + circle
        dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color="white", width=2)),
        dict(type="circle", x0=47-6, y0=25-6, x1=47+6, y1=25+6, line=dict(color="white", width=2)),
    ]

    fig.update_layout(
        shapes=court_shapes,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
    return fig

# -------------------------------
# Streamlit Display Helper
# -------------------------------
def display_charts(charts, df):
    st.subheader("📊 Interactive Performance Visualizations")
    for key, fig in charts.items():
        with st.expander(fig.layout.title.text if fig.layout.title else key, expanded=True):
            st.plotly_chart(fig, use_container_width=True)
            if key == "heatmap":
                st.markdown("**Note:** Heatmap shows average player positions on a half-court basketball overlay.")
            elif key == "radar":
                st.markdown("**Note:** Radar chart compares top 3 players across key metrics.")
            elif key == "distribution":
                st.markdown("**Note:** Box plots show distribution of various performance metrics.")
            elif key == "speed_vs_distance":
                st.markdown("**Note:** Scatter plot visualizes relationship between speed and distance covered.")
            elif key == "avg_speed":
                st.markdown("**Note:** Bar chart ranks players by their average speed.")
    st.markdown("⚠️ Charts are based on uploaded data. Ensure data quality for accurate insights.")
    st.markdown("📥 Upload new data to refresh visualizations.")