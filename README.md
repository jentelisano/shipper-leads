# Freight Radar: Local Shipper Finder

A Streamlit app that discovers and ranks likely **shippers** (manufacturers, distributors, warehouses) for a target city/state.
It pulls data from **Google Places (optional)**, **OpenCorporates**, and **OpenStreetMap/Overpass**, scrapes websites (optional), computes a freight score, and outputs a CSV + interactive map.

## Quick Start

```bash
pip install -r requirements.txt
# Optional: add your Google Places key in .env
cp .env.template .env
streamlit run app.py
```

## Notes
- Google Places dramatically improves coverage/accuracy. Create an API key and enable the Places API.
- Overpass queries may throttle; adjust radius if needed.
- Website scraping is optional (slower) but boosts precision.
- Outputs: CSV file with name, address, lat/lon, website, source, types, freight_score.


## AI Enrichment (Optional)
Set your OpenAI key in the environment:

```bash
export OPENAI_API_KEY=sk-...  # or add to .env
```

In the app, click **AI Enrich (LLM)** to extract company facts, classify to a logistics taxonomy, and compute a lane-fit score for NE → Upstate NY.

