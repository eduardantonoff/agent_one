import json
import streamlit as st
from typing import List, Dict
from dataclasses import dataclass, field
import threading
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize a lock for thread safety
file_lock = threading.Lock()

# Define Data Classes
@dataclass
class Evaluation:
    label: str
    type: str
    value: Dict
    status: str

@dataclass
class Concept:
    id: int
    label: str
    description: str
    status: str
    prerequisite: List[int]
    section: List[str]
    content: List[Dict]
    evaluation: List[Evaluation]

@dataclass
class KST:
    concepts: Dict[int, Concept] = field(default_factory=dict)
    connections: List[Dict] = field(default_factory=list)

    @staticmethod
    def from_json(json_data: Dict) -> 'KST':
        concepts = {}
        for concept in json_data.get('concepts', []):
            evaluations = [Evaluation(**eval_item) for eval_item in concept.get('evaluation', [])]
            concept_obj = Concept(
                id=concept['id'],
                label=concept['label'],
                description=concept['description'],
                status=concept['status'],
                prerequisite=concept['prerequisite'],
                section=concept['section'],
                content=concept['content'],
                evaluation=evaluations
            )
            concepts[concept['id']] = concept_obj
        connections = json_data.get('connections', [])
        return KST(concepts=concepts, connections=connections)

    def to_json(self) -> Dict:
        return {
            "concepts": [
                {
                    "id": concept.id,
                    "label": concept.label,
                    "description": concept.description,
                    "status": concept.status,
                    "prerequisite": concept.prerequisite,
                    "section": concept.section,
                    "content": concept.content,
                    "evaluation": [
                        {
                            "label": eval_item.label,
                            "type": eval_item.type,
                            "value": eval_item.value,
                            "status": eval_item.status
                        } for eval_item in concept.evaluation
                    ]
                } for concept in self.concepts.values()
            ],
            "connections": self.connections  # Ensure connections are included
        }


# Utility Functions for Loading and Saving KST Data
def load_kst_from_file(file_path: str) -> KST:
    with file_lock:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            logger.info(f"Successfully loaded KST data from '{file_path}'.")
            return KST.from_json(data)
        except FileNotFoundError:
            st.error(f"File '{file_path}' not found.")
            logger.error(f"File '{file_path}' not found.")
            return KST()
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from '{file_path}': {e}")
            logger.error(f"Error decoding JSON from '{file_path}': {e}")
            return KST()

def save_kst_to_file(file_path: str, kst: KST) -> None:
    with file_lock:
        try:
            with open(file_path, 'w') as file:
                json.dump(kst.to_json(), file, indent=2)
            logger.info(f"Successfully saved KST data to '{file_path}'.")
        except Exception as e:
            st.error(f"Error saving file '{file_path}': {e}")
            logger.error(f"Error saving file '{file_path}': {e}")

# Define the Reset All Function
def reset_all(file_path: str):
    """
    Resets all concept and evaluation statuses to '-', clears chat history, labels,
    and reinitializes the agent's memory while preserving connections.
    """
    # Load the current KST data
    kst = load_kst_from_file(file_path)
    
    # Reset all concept and evaluation statuses to '-'
    for concept in kst.concepts.values():
        concept.status = "-"
        for eval_item in concept.evaluation:
            eval_item.status = "-"
    
    # Save the updated KST data
    save_kst_to_file(file_path, kst)
    logger.info("All concept and evaluation statuses have been reset to '-'.")
    
    # Clear Streamlit session states for messages and labels
    st.session_state.messages = [
        {
            "role": "Agent",
            "content": (
                "I'm your **assistant**. "
                "My role is to help you understand various statistical concepts, evaluate your knowledge, "
                "and track your progress as you learn.\n\n"
                "Here's how I can assist you:\n\n"
                "- **Learn Concepts**: I can explain statistical concepts in detail, helping you grasp key ideas step by step.\n"
                "- **Practice and Evaluate**: Once you're comfortable, I can provide practice problems and evaluations to test your understanding.\n"
                "- **Track Your Progress**: I'll automatically track your learning progress and adjust your status to reflect mastery as you improve.\n"
                "- **Customized Feedback**: Based on your performance, I'll offer targeted feedback to reinforce learning and guide you to mastery.\n\n"
                "Feel free to ask me anything to get started on your learning journey!"
            )
        }
    ]
    st.session_state.labels = set()
    logger.info("Chat history and labels have been cleared.")
    
    # Reset other session state variables if they exist
    if 'abot' in st.session_state:
        del st.session_state.abot
    if 'thread' in st.session_state:
        del st.session_state.thread
    if 'embed' in st.session_state:
        del st.session_state.embed
    
    # Rerun the app to reflect changes
    st.rerun()