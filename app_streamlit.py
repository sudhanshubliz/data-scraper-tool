# app_streamlit_full.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import re
from itertools import combinations
from typing import List, Dict

# Topic modeling imports
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# Helper: fetch papers from Crossref
# -----------------------------
def fetch_papers(query: str, rows: int = 50) -> List[Dict]:
    url = f"https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("message", {}).get("items", [])
    papers = []
    for it in items:
        title = it.get("title", [""])[0] if it.get("title") else ""
        issued = it.get("issued", {}).get("date-parts", [[None]])
        year = issued[0][0] if issued and issued[0] else None
        doi = it.get("DOI")
        abstract = it.get("abstract")  # may be None or HTML markup
        # normalize abstract: remove XML/HTML tags if present
        if abstract:
            abstract = re.sub(r"<[^>]+>", " ", abstract)
            abstract = re.sub(r"\s+", " ", abstract).strip()
        # authors: try to get list of "Given Family"
        authors = []
        for a in it.get("author", []) or []:
            given = a.get("given") or ""
            family = a.get("family") or ""
            name = (given + " " + family).strip() or a.get("name") or None
            if name:
                authors.append(name)
        # references (may contain DOI)
        references = []
        for r in it.get("reference", []) or []:
            ref_doi = r.get("DOI") or r.get("doi")
            if ref_doi:
                references.append(ref_doi.lower())
        # citation count (Crossref field)
        cited_by_count = it.get("is-referenced-by-count", 0)
        papers.append({
            "title": title,
            "year": year,
            "doi": doi.lower() if doi else None,
            "abstract": abstract,
            "authors": authors,
            "references": references,
            "cited_by_count": cited_by_count
        })
    return papers

# -----------------------------
# Build co-author graph
# -----------------------------
def build_coauthor_graph(papers: List[Dict], min_edge_weight: int = 1) -> nx.Graph:
    G = nx.Graph()
    for p in papers:
        authors = list(dict.fromkeys(p.get("authors", [])))  # preserve unique and order
        for a in authors:
            if not G.has_node(a):
                G.add_node(a, type="author")
        # add edges between all pairs in this paper
        for a1, a2 in combinations(authors, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)
    # optionally remove weak edges / isolated nodes
    if min_edge_weight > 1:
        weak = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_edge_weight]
        G.remove_edges_from(weak)
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
    return G

# -----------------------------
# Build citation graph (best-effort)
# Nodes: DOIs present in fetched set. Edges: paper -> referenced paper (if DOI matched)
# -----------------------------
def build_citation_graph(papers: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    doi_to_title = {}
    # map DOIs to titles for nodes
    for p in papers:
        if p.get("doi"):
            doi_to_title[p["doi"]] = p.get("title") or p["doi"]
            G.add_node(p["doi"], title=p.get("title"))
    # add edges when reference DOI points to another fetched DOI
    for p in papers:
        src = p.get("doi")
        if not src:
            continue
        for ref in p.get("references", []):
            ref_norm = ref.lower()
            if ref_norm in doi_to_title:
                # edge: src cites ref_norm
                G.add_edge(src, ref_norm)
    return G

# -----------------------------
# Simple visualization helpers
# -----------------------------
def plot_coauthor_graph(G: nx.Graph, top_n_authors: int = 30):
    if G.number_of_nodes() == 0:
        st.info("Co-author graph is empty.")
        return
    # pick top nodes by degree
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top = [n for n, d in deg[:top_n_authors]]
    H = G.subgraph(top).copy()
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(H, k=0.5, seed=42)
    weights = [H[u][v].get("weight", 1) for u, v in H.edges()]
    nx.draw_networkx_nodes(H, pos, node_size=[300 + 200 * H.degree(n) for n in H.nodes()])
    nx.draw_networkx_edges(H, pos, width=[0.5 + w for w in weights], alpha=0.7)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.axis("off")
    st.pyplot(plt)

def plot_citation_graph(G: nx.DiGraph, top_n: int = 25):
    if G.number_of_nodes() == 0:
        st.info("Citation graph is empty or no reference DOIs matched.")
        return
    # pick top nodes by in-degree (most cited within fetched set)
    indeg = sorted(G.in_degree, key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, d in indeg[:top_n]]
    H = G.subgraph(top_nodes).copy().to_undirected()
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(H, k=0.5, seed=24)
    nx.draw_networkx_nodes(H, pos, node_size=[300 + 200 * H.degree(n) for n in H.nodes()])
    nx.draw_networkx_edges(H, pos, alpha=0.6)
    # use DOI (shorten) as labels
    labels = {n: (n.split("/")[-1] if "/" in n else n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, font_size=7)
    plt.axis("off")
    st.pyplot(plt)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CNT & CCUS Research Explorer", layout="wide")
st.title("📚 CNT & CCUS Research Explorer — Topics + Networks")

with st.sidebar:
    st.header("Search & Options")
    query = st.text_input("Search keyword", value="Carbon Nanotubes")
    rows = st.number_input("Max papers to fetch", min_value=10, max_value=200, value=60, step=10)
    do_topic = st.checkbox("Run topic modeling (BERTopic)", value=True)
    do_networks = st.checkbox("Build co-author & citation networks", value=True)
    min_edge_weight = st.slider("Min co-author edge weight to keep (filter)", 1, 5, 1)
    run = st.button("Fetch & Analyze")

if run:
    with st.spinner("Fetching papers from Crossref..."):
        try:
            papers = fetch_papers(query, rows=rows)
        except Exception as e:
            st.error(f"Error fetching from Crossref: {e}")
            st.stop()

    if not papers:
        st.warning("No papers returned. Try a different keyword or increase the number of rows.")
        st.stop()

    df = pd.DataFrame(papers)
    st.success(f"Fetched {len(df)} papers")
    # show basic table
    st.subheader("Fetched papers (sample)")
    st.dataframe(df[["title", "year", "doi", "cited_by_count"]].head(100))

    # Publications per year
    st.subheader("Publications per Year")
    year_counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots()
    year_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Top keywords from titles
    st.subheader("Top keywords in titles")
    text = " ".join(df["title"].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-zA-Z0-9\-]+\b", text)
    stopwords = set(["the","and","of","in","to","a","for","on","with","by","using","via"])
    words = [w for w in words if w not in stopwords and len(w) > 2]
    common = Counter(words).most_common(15)
    kw_df = pd.DataFrame(common, columns=["word", "count"])
    st.bar_chart(kw_df.set_index("word"))

    # Topic modeling
    if do_topic:
        st.subheader("Topic Modeling (BERTopic) — abstracts")
        abstracts = df["abstract"].dropna().astype(str).tolist()
        if len(abstracts) < 2:
            st.info("Not enough abstracts to run topic modeling. Try fetching more papers or pick another keyword.")
        else:
            with st.spinner("Fitting BERTopic... (may take ~20-60s depending on rows)"):
                try:
                    vectorizer_model = CountVectorizer(stop_words="english", max_df=0.9, min_df=2)
                    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=False)
                    topics, probs = topic_model.fit_transform(abstracts)
                    topic_info = topic_model.get_topic_info()
                    st.write(topic_info.head(15))
                    # barchart
                    try:
                        fig = topic_model.visualize_barchart(top_n_topics=10)
                        st.plotly_chart(fig)
                    except Exception:
                        st.info("Interactive topic chart not available; relying on topic table above.")
                except Exception as e:
                    st.error(f"BERTopic failed: {e}")

    # Networks
    if do_networks:
        st.subheader("Co-author Collaboration Network")
        G_co = build_coauthor_graph(papers, min_edge_weight=min_edge_weight)
        st.write(f"Nodes: {G_co.number_of_nodes()}, Edges: {G_co.number_of_edges()}")
        plot_coauthor_graph(G_co, top_n_authors=30)

        st.subheader("Citation Network (within fetched set, best-effort)")
        G_cit = build_citation_graph(papers)
        st.write(f"Nodes: {G_cit.number_of_nodes()}, Edges: {G_cit.number_of_edges()}")
        plot_citation_graph(G_cit, top_n=25)

    st.info("Notes: Citation edges only appear when Crossref returned reference DOIs and those DOIs match DOIs present in the fetched set. Co-author graph is built from author metadata in Crossref and counts co-authorship frequency.")

