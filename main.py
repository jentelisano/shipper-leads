
import os
import re
import io
import json
import time
import math
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus
from dotenv import load_dotenv

# Optional map (folium) for visualization
import folium
from folium.plugins import MarkerCluster

load_dotenv()

st.set_page_config(page_title="Freight Radar: Local Shipper Finder", page_icon="🚚", layout="wide")

st.title("🚚 Freight Radar: Local Shipper Finder")
st.write("Generate a **ranked list of shippers** (manufacturers, distributors, warehouses) for a target location.")

# -----------------------
# UI
# -----------------------
with st.sidebar:
    st.header("🔧 Settings")
    city = st.text_input("City", value="Londonderry")
    state = st.text_input("State (e.g., NH)", value="NH")
    country = st.text_input("Country", value="USA")
    search_radius_km = st.slider("Search radius (km)", min_value=5, max_value=50, value=15, step=5)

    st.markdown("---")
    st.caption("**Optional API keys** (improve coverage & accuracy)")
    gmaps_key = st.text_input("Google Places API Key", value=os.getenv("GOOGLE_PLACES_API_KEY", ""))
    # OpenCorporates does not strictly require an API key for basic search
    use_google = st.checkbox("Use Google Places", value=True)
    use_opencorp = st.checkbox("Use OpenCorporates", value=True)
    use_overpass = st.checkbox("Use OpenStreetMap (Overpass)", value=True)

    st.markdown("---")
    scrape_websites = st.checkbox("Scrape company websites for keywords (slower)", value=False)
    run_button = st.button("Run discovery")

# -----------------------
# Helpers
# -----------------------
KEYWORDS = [
    "manufacturing", "distribution", "distributor", "warehouse", "fulfillment",
    "dock", "shipping", "logistics", "pallet", "freight", "truckload", "tl", "ltl",
    "food", "beverage", "packaging", "plastics", "ingredients", "chemicals",
    "building materials", "paper", "printing", "textile", "automotive", "electronics"
]

CATEGORIES = [
    "manufacturer", "distributor", "warehouse", "logistics", "industrial", "plant", "facility"
]

def safe_get(url, params=None, headers=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
        return None
    except Exception:
        return None

def geocode_city(city, state, country="USA"):
    q = f"{city}, {state}, {country}"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    r = safe_get(url, params=params, headers={"User-Agent": "FreightRadar/1.0"})
    if not r:
        return None
    data = r.json()
    if not data:
        return None
    d = data[0]
    return {
        "lat": float(d["lat"]),
        "lon": float(d["lon"]),
        "display_name": d.get("display_name", ""),
        "boundingbox": [float(x) for x in d["boundingbox"]]
    }

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

# -----------------------
# Google Places
# -----------------------
def gp_text_search(query, key, next_page_token=None):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": key}
    if next_page_token:
        params["pagetoken"] = next_page_token
    r = safe_get(url, params=params)
    if not r:
        return []
    return r.json()

def gp_place_details(place_id, key):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "key": key, "fields": "name,formatted_address,geometry,url,website,types"}
    r = safe_get(url, params=params)
    if not r:
        return {}
    return r.json().get("result", {})

def google_places_search(lat, lon, radius_m, key):
    queries = [
        "manufacturing", "warehouse", "distribution center", "food distributor",
        "packaging", "plastics", "printing", "ingredients", "chemicals", "building materials"
    ]
    results = []
    # Use Places Text Search with "in CITY STATE" to bias location
    for term in queries:
        q = f"{term} near {city}, {state}"
        data = gp_text_search(q, key)
        if isinstance(data, dict) and data.get("results"):
            for res in data["results"]:
                place_id = res.get("place_id")
                try:
                    details = gp_place_details(place_id, key)
                except Exception:
                    details = {}
                results.append({
                    "name": details.get("name") or res.get("name"),
                    "address": details.get("formatted_address", ""),
                    "lat": details.get("geometry", {}).get("location", {}).get("lat"),
                    "lon": details.get("geometry", {}).get("location", {}).get("lng"),
                    "website": details.get("website", ""),
                    "source": "google_places",
                    "types": ",".join(details.get("types", []))
                })
        time.sleep(1)  # respect rate limits
    return results

# -----------------------
# OpenCorporates
# -----------------------
def opencorporates_search(city, state):
    # Simple search, then filter by city/state in address if returned
    url = "https://api.opencorporates.com/v0.4/companies/search"
    params = {"q": f"{city} {state}"}
    r = safe_get(url, params=params)
    out = []
    if not r:
        return out
    data = r.json()
    companies = data.get("results", {}).get("companies", [])
    for c in companies:
        comp = c.get("company", {})
        name = comp.get("name", "")
        # Pull address lines if available
        addr = ""
        if comp.get("registered_address"):
            addr = comp.get("registered_address", "")
        elif comp.get("registered_address_in_full"):
            addr = comp.get("registered_address_in_full", "")
        out.append({
            "name": name,
            "address": addr,
            "lat": None,
            "lon": None,
            "website": "",
            "source": "opencorporates",
            "types": ""
        })
    return out

# -----------------------
# Overpass (OpenStreetMap) for industrial/warehouse POIs
# -----------------------
def overpass_query(lat, lon, radius_m):
    # Look for landuse=industrial, building=warehouse, and industrial/manufacturing amenities
    # within a radius of the city centroid.
    q = f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})[landuse=industrial];
      way(around:{radius_m},{lat},{lon})[landuse=industrial];
      relation(around:{radius_m},{lat},{lon})[landuse=industrial];

      node(around:{radius_m},{lat},{lon})[building=warehouse];
      way(around:{radius_m},{lat},{lon})[building=warehouse];
      relation(around:{radius_m},{lat},{lon})[building=warehouse];

      node(around:{radius_m},{lat},{lon})[industrial];
      way(around:{radius_m},{lat},{lon})[industrial];
      relation(around:{radius_m},{lat},{lon})[industrial];
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    r = safe_get(url, data=q, headers={"Content-Type": "text/plain", "User-Agent": "FreightRadar/1.0"})
    if not r:
        return []
    data = r.json()
    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            # skip unnamed
            continue
        lat_, lon_ = None, None
        if "lat" in el and "lon" in el:
            lat_, lon_ = el["lat"], el["lon"]
        elif "center" in el:
            lat_, lon_ = el["center"]["lat"], el["center"]["lon"]
        out.append({
            "name": name,
            "address": "",
            "lat": lat_,
            "lon": lon_,
            "website": "",
            "source": "overpass_osm",
            "types": ",".join([k+"="+v for k,v in tags.items() if k in ("landuse","building","industrial","man_made","office")])
        })
    return out

# -----------------------
# Enrichment
# -----------------------
def guess_category(text):
    text = text.lower()
    score = 0
    for kw in KEYWORDS:
        if kw in text:
            score += 1
    return score

def scrape_website(url):
    try:
        r = safe_get(url, headers={"User-Agent": "FreightRadar/1.0"}, timeout=15)
        if not r:
            return ""
        # basic clean
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)[:20000]  # cap size
        return text
    except Exception:
        return ""

def enrich_rows(rows, scrape=False):
    df = pd.DataFrame(rows).fillna("")
    # Deduplicate by name+address
    if not df.empty:
        df = df.drop_duplicates(subset=["name","address"], keep="first")
    # Basic keyword scoring on types (and optional website scrape)
    scores = []
    for i, r in df.iterrows():
        base_text = (r.get("types","") or "") + " " + (r.get("address","") or "") + " " + (r.get("name","") or "")
        s = guess_category(base_text)
        web_text = ""
        if scrape and r.get("website"):
            web_text = scrape_website(r["website"])
            s += min(10, guess_category(web_text))  # cap contribution
        scores.append(s)
    df["freight_score"] = scores
    # Heuristic: filter out low-signal rows
    df = df[df["freight_score"] > 0].copy()
    # Sort
    df = df.sort_values("freight_score", ascending=False)
    return df

# -----------------------
# Run
# -----------------------
if run_button:
    with st.spinner("Geocoding city..."):
        geo = geocode_city(city, state, country)
    if not geo:
        st.error("Could not geocode the city/state. Check spelling and try again.")
        st.stop()
    lat, lon = geo["lat"], geo["lon"]
    st.success(f"Geocoded: {geo['display_name']} (lat {lat:.4f}, lon {lon:.4f})")

    all_rows = []

    if use_google and gmaps_key.strip():
        st.info("Querying Google Places…")
        rows = google_places_search(lat, lon, int(search_radius_km*1000), gmaps_key.strip())
        st.write(f"Google Places results: {len(rows)}")
        all_rows.extend(rows)
    elif use_google and not gmaps_key.strip():
        st.warning("Google Places selected but no API key provided — skipping.")

    if use_opencorp:
        st.info("Querying OpenCorporates…")
        rows = opencorporates_search(city, state)
        st.write(f"OpenCorporates results: {len(rows)}")
        all_rows.extend(rows)

    if use_overpass:
        st.info("Querying OpenStreetMap/Overpass (industrial/warehouse)…")
        rows = overpass_query(lat, lon, int(search_radius_km*1000))
        st.write(f"Overpass results: {len(rows)}")
        all_rows.extend(rows)

    if not all_rows:
        st.error("No results returned from selected sources. Try enabling more sources, increasing radius, or adding a Google API key.")
        st.stop()

    st.info("Enriching & scoring results…")
    if ai_button:
        with st.spinner("Running AI enrichment..."):
            df_enriched = run_ai_enrich(df, top_n=ai_top_n)
        st.subheader("🧠 AI-Enriched Results")
        st.dataframe(df_enriched, use_container_width=True)
        csv_ai = df_enriched.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download AI-Enriched CSV", data=csv_ai, file_name=f"shippers_{city}_{state}_AI.csv", mime="text/csv")

    df = enrich_rows(all_rows, scrape=scrape_websites)

    st.subheader("📋 Candidate Shippers")
    st.caption("Sorted by freight score (higher = more likely shipper).")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"shippers_{city}_{state}.csv", mime="text/csv")

    # Map

    st.markdown("---")
    st.subheader("🤖 AI Enrichment")
    st.caption("Optional: Use OpenAI to extract & classify facilities, then compute a lane-fit score. Set OPENAI_API_KEY in your environment.")
    st.session_state["openai_model"] = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4o-mini-2024-07-18"], index=0)
    ai_top_n = st.slider("Number of rows to enrich", 10, 200, 100, 10)
    ai_button = st.button("AI Enrich (LLM)")

    st.subheader("🗺️ Map")
    m = folium.Map(location=[lat, lon], zoom_start=11, control_scale=True)
    mc = MarkerCluster().add_to(m)

    for _, r in df.iterrows():
        if r.get("lat") and r.get("lon"):
            popup = f"<b>{r['name']}</b><br>{r.get('address','')}<br>Score: {r.get('freight_score','')}<br>Source: {r.get('source','')}"
            if r.get("website"):
                popup += f"<br><a href='{r['website']}' target='_blank'>Website</a>"
            folium.Marker(
                [float(r["lat"]), float(r["lon"])],
                popup=popup,
                tooltip=r["name"],
            ).add_to(mc)

    # Render map
    from streamlit_folium import st_folium
    st_folium(m, width=1200, height=600)

    st.success("Done! Tip: Toggle website scraping to boost precision (slower).")

else:
    st.caption("Enter a city/state, choose data sources, and click **Run discovery**. Add a Google Places key for best results.")


# =======================
# AI Enrichment (OpenAI)
# =======================
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

class ShippingSignals(BaseModel):
    has_docks: bool = False
    ships_ftl: bool = False
    ships_ltl: bool = False
    has_private_fleet: bool = False
    accepts_appointments: bool = False
    has_edi: bool = False
    exports_imports: bool = False

class ExtractedCompany(BaseModel):
    name: str = ""
    website: Optional[str] = None
    address_text: Optional[str] = None
    phone: Optional[str] = None
    emails: List[str] = []
    keywords: List[str] = []
    summary: Optional[str] = None
    shipping_signals: ShippingSignals = ShippingSignals()

class Classification(BaseModel):
    facility_type: str = ""       # manufacturer | distributor | warehouse | 3PL | DC | office_only | retail
    commodities: List[str] = []   # e.g., food_bev, packaging, plastics...
    naics_candidates: List[str] = []
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    lane_fit_NE_to_UpstateNY: bool = False
    lane_fit_reason: str = ""

def _openai_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
    except Exception:
        return None

def _clean_text(t: str, max_chars: int = 20000) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t[:max_chars]

def score_company(extr: ExtractedCompany, cls: Classification) -> Dict[str, Any]:
    score = 0
    score += {"manufacturer":6, "distributor":5, "warehouse":4, "DC":4, "3PL":3}.get(cls.facility_type, 0)
    s = extr.shipping_signals
    score += 2 if s.has_docks else 0
    score += 2 if s.ships_ftl else 0
    score += 1 if s.ships_ltl else 0
    score += 1 if s.has_edi else 0
    heavy = {"building_materials","paper_printing","packaging","beverage","chemicals"}
    score += 2 if any(c in heavy for c in cls.commodities) else 0
    lane_fit = 3 if cls.lane_fit_NE_to_UpstateNY else 0
    total = min(10, score + lane_fit)
    return {"shipper_likelihood": min(10, score), "lane_fit": lane_fit, "total_score": total}

def llm_extract(client, html_text: str) -> Optional[ExtractedCompany]:
    if client is None or not OPENAI_KEY:
        return None
    sys = "You are a logistics analyst. Return ONLY valid JSON for the ExtractedCompany schema. Be conservative."
    usr = f"HTML/TEXT:\n{_clean_text(html_text)}"
    try:
        resp = client.chat.completions.create(
            model=st.session_state.get("openai_model","gpt-4o-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            response_format={"type":"json_object"},
            temperature=0.1,
        )
        js = resp.choices[0].message.content
        data = json.loads(js)
        return ExtractedCompany(**data)
    except Exception as e:
        return None

def llm_classify(client, extracted: ExtractedCompany) -> Optional[Classification]:
    if client is None or not OPENAI_KEY:
        return None
    taxonomy = """
Facility types: manufacturer | distributor | warehouse | 3PL | DC | office_only | retail
Commodities: food_bev, packaging, plastics, chemicals, ingredients, paper_printing, building_materials, automotive, electronics, pharma, household, industrial_supplies
Return JSON only per Classification schema.
"""
    usr = f"Extracted facts:\n{extracted.model_dump_json()}"
    try:
        resp = client.chat.completions.create(
            model=st.session_state.get("openai_model","gpt-4o-mini"),
            messages=[{"role":"system","content":taxonomy},{"role":"user","content":usr}],
            response_format={"type":"json_object"},
            temperature=0.1,
        )
        js = resp.choices[0].message.content
        data = json.loads(js)
        return Classification(**data)
    except Exception:
        return None

def fetch_text_from_site(url: str) -> str:
    if not url:
        return ""
    try:
        r = safe_get(url, headers={"User-Agent":"FreightRadar/AIEnrich"} , timeout=15)
        if not r:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def run_ai_enrich(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    client = _openai_client()
    if client is None:
        st.warning("OpenAI client not available. Set OPENAI_API_KEY in your environment.")
        return df

    work = df.head(top_n).copy()
    extr_list, cls_list, score_list = [], [], []
    for i, row in work.iterrows():
        html_text = ""
        if row.get("website"):
            html_text = fetch_text_from_site(str(row["website"]))
        extracted = llm_extract(client, html_text) if html_text else ExtractedCompany(name=row.get("name",""), website=row.get("website",""))
        if not extracted:
            extracted = ExtractedCompany(name=row.get("name",""), website=row.get("website",""))
        classified = llm_classify(client, extracted) or Classification()

        scores = score_company(extracted, classified)
        extr_list.append(extracted.model_dump())
        cls_list.append(classified.model_dump())
        score_list.append(scores)

    work["ai_extraction"] = extr_list
    work["ai_classification"] = cls_list
    work["ai_scores"] = score_list
    work["ai_total_score"] = [s["total_score"] for s in score_list]

    # Merge AI results back to original df
    df_out = df.merge(work[["name","address","ai_extraction","ai_classification","ai_scores","ai_total_score"]],
                      on=["name","address"], how="left")
    return df_out
