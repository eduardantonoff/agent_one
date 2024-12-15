import pandas as pd
import streamlit as st
from typing import List, Dict
from dataclasses import dataclass, field
import os

# Import shared utilities
from utils import load_kst_from_file, save_kst_to_file, reset_all, KST, Concept, Evaluation

# Define File Path
FILE_PATH = 'data/schema.json'

# Verify File Exists
if not os.path.exists(FILE_PATH):
    st.error(f"The file '{FILE_PATH}' does not exist. Please ensure the file is in the correct directory.")
    st.stop()

# Load KST Data
kst = load_kst_from_file(FILE_PATH)

# Generate Status Vector
def generate_concepts_status_vector(kst: KST) -> pd.DataFrame:
    """
    Generates a DataFrame for concepts with their statuses.
    """
    sorted_concepts = sorted(kst.concepts.values(), key=lambda c: c.id)
    data = {
        'Concept': [concept.label.split(' ', 1)[-1] for concept in sorted_concepts],
        'Status': [concept.status.strip() for concept in sorted_concepts]
    }
    indices = [concept.label.split(' ', 1)[0] for concept in sorted_concepts]
    df = pd.DataFrame(data, index=indices)
    return df

concepts_status_df = generate_concepts_status_vector(kst)

# Display Status Table
# st.header("Status Overview")
st.table(concepts_status_df)

# Add Reset Button on This Page
with st.form("reset_all_form"):
    reset_button = st.form_submit_button("Reset")
    st.caption("Use this to **reset** all progress data and clear chat history.")

if reset_button:
    reset_all(FILE_PATH)
