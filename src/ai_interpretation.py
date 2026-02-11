"""
ai_interpretation.py
--------------------
AI-powered chart interpretation using OpenAI GPT-5.2 vision with
Pydantic structured output.

Provides:
    - Pydantic models for structured chart analysis results
    - Vision-based chart interpretation via OpenAI's GPT-5.2 model
    - Streamlit rendering of interpretation results
"""

from __future__ import annotations

import base64
import json
import os
from typing import Literal

import openai
from pydantic import BaseModel, ConfigDict
import streamlit as st


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrendInfo(BaseModel):
    """Describes the overall trend detected in the chart."""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["upward", "downward", "flat", "mixed"]
    description: str


class SeasonalityInfo(BaseModel):
    """Describes any seasonality detected in the chart."""

    model_config = ConfigDict(extra="forbid")

    detected: bool
    period: str | None
    description: str


class StationarityInfo(BaseModel):
    """Describes whether the series appears stationary."""

    model_config = ConfigDict(extra="forbid")

    likely_stationary: bool
    description: str


class AnomalyItem(BaseModel):
    """A single anomaly or outlier observation."""

    model_config = ConfigDict(extra="forbid")

    approximate_location: str
    description: str
    severity: Literal["low", "medium", "high"]


class ChartInterpretation(BaseModel):
    """Complete structured interpretation of a time-series chart."""

    model_config = ConfigDict(extra="forbid")

    chart_type_detected: str
    trend: TrendInfo
    seasonality: SeasonalityInfo
    stationarity: StationarityInfo
    anomalies: list[AnomalyItem]
    key_observations: list[str]
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

def check_api_key_available() -> bool:
    """Return ``True`` if the ``OPENAI_API_KEY`` environment variable is set
    and non-empty."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key.strip())


# ---------------------------------------------------------------------------
# Chart interpretation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a careful time-series analyst helping business analytics "
    "students. Analyze the chart image and provide a structured "
    "interpretation. Be precise about what the data shows; flag anything "
    "noteworthy. Use plain language suitable for students."
)


def interpret_chart(
    png_bytes: bytes,
    metadata: dict,
) -> ChartInterpretation:
    """Send a chart image to GPT-5.2 vision and return a structured
    interpretation.

    Parameters
    ----------
    png_bytes:
        Raw PNG image bytes of the chart to analyse.
    metadata:
        Context about the chart.  Expected keys:

        * ``chart_type`` -- e.g. ``"line"``, ``"bar"``, ``"decomposition"``
        * ``frequency_label`` -- e.g. ``"Monthly"``, ``"Daily"``
        * ``date_range`` -- human-readable date range string
        * ``y_column`` -- name of the value column being plotted
    """
    try:
        client = openai.OpenAI()

        # Encode the PNG as a base64 data URI
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{b64}"

        chart_type = metadata.get("chart_type", "time-series")
        metadata_str = json.dumps(metadata, default=str)

        response = client.beta.chat.completions.parse(
            model="gpt-5.2-2025-12-11",
            response_format=ChartInterpretation,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this {chart_type} chart. "
                                f"Metadata: {metadata_str}"
                            ),
                        },
                    ],
                },
            ],
        )

        # Prefer the parsed structured output
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed

        # Fallback: try to manually parse the raw content
        raw_content = response.choices[0].message.content or ""
        data = json.loads(raw_content)
        return ChartInterpretation(**data)

    except Exception as exc:  # noqa: BLE001
        # Return a minimal interpretation that surfaces the error
        return ChartInterpretation(
            chart_type_detected="unknown",
            trend=TrendInfo(direction="mixed", description="Unable to determine."),
            seasonality=SeasonalityInfo(
                detected=False, period=None, description="Unable to determine."
            ),
            stationarity=StationarityInfo(
                likely_stationary=False, description="Unable to determine."
            ),
            anomalies=[],
            key_observations=["AI interpretation failed; see summary for details."],
            summary=f"Error during AI interpretation: {exc}",
            recommendations=["Check that your OPENAI_API_KEY is set and valid."],
        )


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

_DIRECTION_EMOJI = {
    "upward": "\u2197\ufe0f",    # arrow upper-right
    "downward": "\u2198\ufe0f",  # arrow lower-right
    "flat": "\u27a1\ufe0f",      # arrow right
    "mixed": "\u2194\ufe0f",     # left-right arrow
}

_SEVERITY_COLOR = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}


def render_interpretation(interp: ChartInterpretation) -> None:
    """Render a :class:`ChartInterpretation` as a styled Streamlit card.

    Uses ``st.markdown``, ``st.expander``, and related widgets to lay out
    the interpretation in an easy-to-read format with sections for trend,
    seasonality, stationarity, anomalies, key observations, summary, and
    recommendations.
    """

    st.markdown("### AI Chart Interpretation")
    st.markdown(
        f"**Detected chart type:** {interp.chart_type_detected}"
    )

    # ---- Summary ----------------------------------------------------------
    st.markdown("---")
    st.markdown(f"**Summary:** {interp.summary}")

    # ---- Key observations -------------------------------------------------
    with st.expander("Key Observations", expanded=True):
        for obs in interp.key_observations:
            st.markdown(f"- {obs}")

    # ---- Trend ------------------------------------------------------------
    with st.expander("Trend Analysis"):
        arrow = _DIRECTION_EMOJI.get(interp.trend.direction, "")
        st.markdown(
            f"**Direction:** {interp.trend.direction.capitalize()} {arrow}"
        )
        st.markdown(interp.trend.description)

    # ---- Seasonality ------------------------------------------------------
    with st.expander("Seasonality"):
        status = "Detected" if interp.seasonality.detected else "Not detected"
        st.markdown(f"**Status:** {status}")
        if interp.seasonality.period:
            st.markdown(f"**Period:** {interp.seasonality.period}")
        st.markdown(interp.seasonality.description)

    # ---- Stationarity -----------------------------------------------------
    with st.expander("Stationarity"):
        label = (
            "Likely stationary"
            if interp.stationarity.likely_stationary
            else "Likely non-stationary"
        )
        st.markdown(f"**Assessment:** {label}")
        st.markdown(interp.stationarity.description)

    # ---- Anomalies --------------------------------------------------------
    with st.expander("Anomalies"):
        if not interp.anomalies:
            st.markdown("No anomalies detected.")
        else:
            for anomaly in interp.anomalies:
                color = _SEVERITY_COLOR.get(anomaly.severity, "gray")
                st.markdown(
                    f"- **[{anomaly.approximate_location}]** "
                    f":{color}[{anomaly.severity.upper()}] "
                    f"-- {anomaly.description}"
                )

    # ---- Recommendations --------------------------------------------------
    with st.expander("Recommended Next Steps"):
        for rec in interp.recommendations:
            st.markdown(f"1. {rec}")
