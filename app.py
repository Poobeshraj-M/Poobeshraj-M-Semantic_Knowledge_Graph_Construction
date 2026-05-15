import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Set page configuration for a premium feel
st.set_page_config(
    page_title="Semantic Knowledge Graph Pro",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .header-style {
        font-size: 40px;
        font-weight: bold;
        background: -webkit-linear-gradient(#4A90E2, #141E30);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Model Loading (Safe CPU mode)
# -----------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return nlp, model

nlp, model = load_models()

# -----------------------------
# Utility: Text Normalization
# -----------------------------
def normalize(text):
    return text.lower().replace("the ", "").strip()

# -----------------------------
# NLP Processing Functions
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_relations(text):
    doc = nlp(text)
    relations = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = [w.text for w in token.lefts if w.dep_ in ["nsubj", "nsubjpass"]]
            obj = [w.text for w in token.rights if w.dep_ in ["dobj", "attr", "dative", "oprd"]]
            if subject and obj:
                relations.append((subject[0], token.text, obj[0]))
    return relations

# -----------------------------
# Step 3: Domain Linking (Semantic Similarity)
# -----------------------------
def link_domains(triples_df, threshold=0.6):
    # Create full semantic phrases from triples
    triples_df["sentence_rep"] = triples_df.apply(
        lambda row: f"{row['Entity1']} {row['Relation']} {row['Entity2']}", axis=1
    )
    sentences = triples_df["sentence_rep"].drop_duplicates().tolist()

    if len(sentences) < 2:
        return []

    embeddings = model.encode(sentences, convert_to_tensor=True)
    linked = []

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if score > threshold:
                linked.append((sentences[i], sentences[j], round(score, 3)))

    return sorted(linked, key=lambda x: x[2], reverse=True)[:10]

# -----------------------------
# Visualization & Analytics
# -----------------------------
def visualize_knowledge_graph(triples_df, highlight_nodes=None):
    G = nx.DiGraph()
    for _, row in triples_df.iterrows():
        G.add_node(row["Entity1"])
        G.add_node(row["Entity2"])
        G.add_edge(row["Entity1"], row["Entity2"], label=row["Relation"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Graph Analytics")
        if len(G.nodes) > 0:
            # Centrality
            degree_centrality = nx.degree_centrality(G)
            top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            with st.expander("Key Influencers (Centrality)", expanded=True):
                for node, score in top_central_nodes:
                    st.write(f"**{node}** : `{score:.3f}`")

            # Communities
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G))
                with st.expander(f"Detected {len(communities)} Communities"):
                    for i, community in enumerate(communities):
                        st.write(f"**Group {i+1}:** {', '.join(list(community)[:10])}...")
            except Exception as e:
                st.info("Community detection skipped (Insufficient data).")
        else:
            st.warning("No data for analytics.")

    with col2:
        st.subheader("🌐 Interactive Map")
        net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
        
        # Add nodes with custom styling
        for node in G.nodes:
            color = "#FF4B4B" if highlight_nodes and node in highlight_nodes else "#4A90E2"
            size = 25 if highlight_nodes and node in highlight_nodes else 15
            net.add_node(node, label=node, color=color, size=size)
        
        # Add edges
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], label=edge[2]["label"], color="#888888")
        
        # Configure physics for better layout
        net.toggle_physics(True)
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          }
        }
        """)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.write_html(tmp_file.name)
            components.html(open(tmp_file.name, "r").read(), height=600)

# -----------------------------
# Main Application UI
# -----------------------------
def main():
    st.markdown('<div class="header-style">Semantic Knowledge Graph Pro</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings & Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1]
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "sentence" not in df.columns:
            st.error("Error: The file must contain a 'sentence' column.")
            return

        # Extraction logic
        with st.spinner("🧠 Extracting knowledge..."):
            triples = []
            for text in df["sentence"].dropna():
                triples.extend(extract_relations(text))
            
            triples_df = pd.DataFrame(triples, columns=["Entity1", "Relation", "Entity2"])
            
        # UI Layout with Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Knowledge View", "🔍 Intelligent Search", "🔗 Domain Insights"])

        with tab1:
            st.write("### Data Explorer")
            st.dataframe(triples_df, use_container_width=True, height=300)
            
            # Download option
            csv = triples_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Extracted Triples (CSV)",
                data=csv,
                file_name="extracted_knowledge.csv",
                mime="text/csv",
            )
            
            st.divider()
            visualize_knowledge_graph(triples_df)

        with tab2:
            st.write("### Smart Search & Query")
            
            search_col, qa_col = st.columns(2)
            
            with search_col:
                with st.container(border=True):
                    st.subheader("🔍 Node Search")
                    query = st.text_input("Find an entity (Semantic Search):", key="node_search")
                    highlight_nodes = None
                    if query:
                        all_nodes = list(set(triples_df["Entity1"].tolist() + triples_df["Entity2"].tolist()))
                        node_embeddings = model.encode(all_nodes, convert_to_tensor=True)
                        query_embedding = model.encode(query, convert_to_tensor=True)
                        cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
                        results = sorted(zip(all_nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:5]

                        for node, score in results:
                            st.write(f"- **{node}** (Similarity: `{score:.3f}`)")
                        highlight_nodes = [r[0] for r in results]

            with qa_col:
                with st.container(border=True):
                    st.subheader("❓ Natural Language QA")
                    question = st.text_input("Ask a question about the data:", key="qa_search")
                    if question:
                        q_doc = nlp(question)
                        q_ents = [ent.text.lower() for ent in q_doc.ents]
                        q_tokens = [token.text.lower() for token in q_doc if token.pos_ in ["NOUN", "PROPN"]]
                        
                        matches = []
                        for _, row in triples_df.iterrows():
                            if any(term in row["Entity1"].lower() or term in row["Entity2"].lower() 
                                   for term in q_ents + q_tokens):
                                matches.append(f"**{row['Entity1']}** {row['Relation']} **{row['Entity2']}**")
                        
                        if matches:
                            st.success("Found matching information:")
                            for m in list(set(matches))[:5]:
                                st.write(f"👉 {m}")
                        else:
                            st.warning("No direct answers found. Try searching for specific entities.")
                
            if highlight_nodes:
                st.info("Searching... The graph below is now highlighting the best matches in RED.")
                visualize_knowledge_graph(triples_df, highlight_nodes=highlight_nodes)

        with tab3:
            st.write("### Semantic Domain Linking")
            st.info("Discover relationships between different knowledge clusters based on semantic similarity.")
            
            domain_links = link_domains(triples_df)
            if domain_links:
                link_df = pd.DataFrame(domain_links, columns=["Concept A", "Concept B", "Similarity"])
                st.table(link_df)
            else:
                st.write("No strong conceptual links found yet. Try uploading more data.")

    else:
        # Welcome Screen
        st.info("👈 Please upload a CSV or Excel file to begin.")
        st.markdown("""
        ### Features:
        - **Automatic Triple Extraction**: NER and Dependency Parsing.
        - **Graph Analytics**: Identify key entities and communities.
        - **Semantic Search**: Find nodes even with different wording.
        - **Interactive Visualization**: Explore the knowledge map dynamically.
        """)
        
        # Show sample if available
        if os.path.exists("sample_data.csv"):
            if st.button("Use Sample Data"):
                df_sample = pd.read_csv("sample_data.csv")
                # Trigger a refresh or just process here
                st.session_state["df_override"] = df_sample
                st.rerun()

if __name__ == "__main__":
    main()
