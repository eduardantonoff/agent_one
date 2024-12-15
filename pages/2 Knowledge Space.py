import streamlit as st
import threading
from utils import load_kst_from_file, KST

# Initialize a lock for thread safety
file_lock = threading.Lock()

def display_concept_details(kst: KST, concept_id: int):
    """
    Displays detailed information about a specific concept, including its evaluations.
    """
    concept = kst.concepts.get(concept_id)
    if not concept:
        st.warning(f"Concept with ID {concept_id} not found.")
        return
    
    # st.header(concept.label)
    st.markdown("**Description:**")
    st.write(concept.description)
    
    st.markdown("**Content:**")
    for item in concept.content:
        if item['type'] == 'instruction':
            st.markdown(f"**Instruction:** {item['value']}")
        elif item['type'] == 'equation':
            st.latex(item['value'])
    
    st.markdown("**Evaluations:**")
    for eval_item in concept.evaluation:
        st.caption(f":blue[**{eval_item.label}:**]")
        if eval_item.type == 'multi':
            st.write(eval_item.value['question'])
            # Assuming multiple-choice options are provided in 'instruction'
            st.write(eval_item.value['instruction'])
        elif eval_item.type == 'problem':
            st.write(eval_item.value['question'])
            st.code(eval_item.value.get('reference', ''))
        elif eval_item.type == 'open':
            st.write(eval_item.value['question'])
            st.write(eval_item.value.get('instruction', ''))
        st.caption(f"Status: {eval_item.status}")
        st.markdown("---")

# Load KST Data
kst = load_kst_from_file('data/schema.json')

# Prepare Concept Selection
concept_ids = sorted(kst.concepts.keys())
concept_labels = [concept.label for concept in sorted(kst.concepts.values(), key=lambda c: c.id)]
concept_dict = {concept.label: concept.id for concept in sorted(kst.concepts.values(), key=lambda c: c.id)}

# Select Concept
selected_label = st.selectbox("Select a Concept to View Details", options=list(concept_dict.keys()))
selected_concept_id = concept_dict[selected_label]

# Display Concept Details
if selected_concept_id:
    display_concept_details(kst, selected_concept_id)