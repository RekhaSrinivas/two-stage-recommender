"""Streamlit demo UI for the two-stage recommender — talks to the FastAPI service.

Run:  streamlit run serving/streamlit_app.py     (with the API up; set API_URL if not local)
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Two-Stage Recommender", page_icon="🎬")
st.title("🎬 Two-Stage Recommender")
st.caption("Two-tower retrieval → LightGCN graph features → LambdaMART ranking")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"Connected — dataset **{health['dataset']}**, {health['n_users']:,} users")
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach API at {API_URL}. Start it first (`docker compose -f "
             f"docker-compose.serving.yaml up`).  [{e}]")
    st.stop()

col1, col2 = st.columns([2, 1])
user_id = col1.number_input("User id", min_value=1, value=1, step=1)
n = col2.slider("How many", 1, 25, 10)

if st.button("Recommend", type="primary"):
    r = requests.get(f"{API_URL}/recommend/{int(user_id)}", params={"n": n}, timeout=30)
    if r.status_code == 404:
        st.warning(f"Unknown user id {int(user_id)}.")
    else:
        hist = requests.get(f"{API_URL}/users/{int(user_id)}/history", timeout=10).json()["history"]
        left, right = st.columns(2)
        left.subheader("Recently liked")
        left.table([{"title": h["title"]} for h in hist])
        right.subheader(f"Top {n} recommendations")
        right.table([{"#": x["rank"], "title": x["title"], "score": round(x["score"], 3)}
                     for x in r.json()["recommendations"]])
