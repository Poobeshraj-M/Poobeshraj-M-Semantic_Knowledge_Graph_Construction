# Semantic Knowledge Graph Construction

An interactive, AI-powered application to extract knowledge triples (Entity-Relation-Entity) from text and visualize them as a dynamic knowledge graph.

## Features
- **Triple Extraction**: Automated NER and Dependency Parsing using Spacy.
- **Interactive Graph**: Visual representation of relationships using PyVis.
- **Graph Analytics**: Insights into central nodes and community clusters.
- **Semantic Search**: Find entities using natural language similarity.
- **Query Answering**: Ask questions about your data and get direct answers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Poobeshraj-M/Poobeshraj-M-Semantic_Knowledge_Graph_Construction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
Upload a CSV or Excel file with a column named `sentence` to begin constructing your semantic map.
