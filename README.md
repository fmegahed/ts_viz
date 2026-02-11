---
title: Time Series Visualizer
emoji: 📈
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Time Series Visualizer + AI Chart Interpreter

A Streamlit app for Miami University Business Analytics students to upload CSV
time-series data, create publication-quality charts, and get AI-powered chart
interpretation.

## Features

- **Upload & Clean** — auto-detect delimiters, date columns, and numeric formats
- **9+ Chart Types** — line, seasonal, subseries, ACF/PACF, decomposition, rolling, YoY, lag, spaghetti
- **Multi-Series Support** — panel (small-multiples) and spaghetti plots for comparing series
- **AI Interpretation** — GPT-5.2 vision analyzes chart images and returns structured insights
- **QueryChat** — natural-language data filtering powered by DuckDB

## Privacy

All data processing happens in-memory. No data is persisted to disk.
Only chart PNG images (never raw data) are sent to the AI when you click "Interpret."
