import os
import re
import uuid
import asyncio
import operator
import logging
import pandas as pd
from typing import List, Dict
from typing import TypedDict, Annotated
import tiktoken

# Streamlit:
import streamlit as st

# LangChain:
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, ToolMessage, trim_messages, get_buffer_string, filter_messages

# LangGraph:
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv

# utils:
from prompts import prompt
from utils import load_kst_from_file, save_kst_to_file


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('.env', override=True)

proxy_api = os.getenv("PROXY_API_KEY")
langsmith_api = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agent One | 4o" # Agent One 4o | Agent One o1

embeddings = OpenAIEmbeddings(api_key=proxy_api,base_url="https://api.proxyapi.ru/openai/v1", model="text-embedding-3-small")
recall_vector_store = InMemoryVectorStore(embeddings)
tokenizer = tiktoken.encoding_for_model("o1-mini")

if 'embed' not in st.session_state:
    st.session_state.embed = recall_vector_store

# Path to KST JSON data
FILE_PATH = 'data/schema.json'

# Define Tools
@tool
def save_recall_memory(memory: str) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    document = Document(page_content=memory, id=str(uuid.uuid4()))
    st.session_state.embed.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str) -> List[str]:
    """Search for relevant memories."""
    documents = st.session_state.embed.similarity_search(query, k=3)
    return [document.page_content for document in documents]

@tool
def retrieve_concepts(query: str) -> str:
    """
    Retrieve concepts based on a section letter or a specific concept ID.
    If retrieved by ID, automatically update status to 'awareness'.
    Do not use large headers (e.g., `#`, `##`, `###`) for titles or subtitles or retrieved content.
    Adapt style and content based on user's preferenfes

    Args:
        query (str): The section letter or concept ID as a string.

    Returns:
        str: Formatted string containing the requested information with a prefix indicating the query type.
    """
    query = query.strip()
    logger.info(f"Received query: '{query}'")
    kst = load_kst_from_file(FILE_PATH)
    concepts = kst.concepts

    # Check if the query is a digit (ID-based query)
    if query.isdigit():
        node_id = int(query)
        concept = concepts.get(node_id)
        if concept:
            label = concept.label
            description = concept.description
            content = concept.content

            # Debugging: Show current status
            logger.info(f"Current status of concept ID {node_id}: '{concept.status}'")

            # Update status to 'awareness' if it's currently '-'
            if concept.status.strip() == '-':
                concept.status = 'awareness'
                save_kst_to_file(FILE_PATH, kst)
                logger.info(f"Updated status of concept ID {node_id} to 'awareness'.")
            else:
                logger.info(f"No status update needed for concept ID {node_id}.")

            # Format the content for readability
            content_formatted = ""
            for item in content:
                if item['type'] == 'instruction':
                    content_formatted += f"**Instruction:** {item['value']}\n\n"
                elif item['type'] == 'equation':
                    content_formatted += f"**Equation:** {item['value']}\n\n"

            # Prefix to indicate ID-based query
            return (f"**ID_QUERY:**\n**ID:** {node_id}\n**Label:** {label}\n"
                    f"**Description:** {description}\n**Content:**\n{content_formatted} NOTE: Use this as a starting point to design best learning material for the user based on hie preferences.")
        else:
            logger.warning(f"No concept found with ID '{node_id}'.")
            return f"No concept found with ID '{node_id}'."

    # Check if the query is a single letter (Section-based query)
    elif re.fullmatch(r'[A-Za-z]', query):
        tag = query.upper()
        logger.info(f"Retrieving concepts for section '{tag}'")
        matching_concepts = [
            f"**SECTION_QUERY:** **ID:** {concept.id}, **Label:** {concept.label}"
            for concept in concepts.values()
            if tag in concept.section
        ]

        if matching_concepts:
            return "\n".join(matching_concepts)
        else:
            logger.warning(f"No concepts found with section '{tag}'.")
            return f"No concepts found with section '{tag}'."

    else:
        logger.warning("Invalid query format.")
        return "Invalid query format. Please enter a single letter for section-based queries or a number for ID-based queries (e.g., '1')."

@tool
def retrieve_evaluations(query: str) -> str:
    """
    Retrieve evaluations for a specific concept.
    Do not use large headers (e.g., `#`, `##`, `###`) for titles or subtitles or retrieved content.
    Adapt style and content based on user's preferenfes

    Args:
        query (str): The concept ID as a string.

    Returns:
        str: Formatted string containing all evaluations for the specified concept.
    """
    query = query.strip()
    kst = load_kst_from_file(FILE_PATH)
    concepts = kst.concepts

    if not query.isdigit():
        return "Invalid query format. Please enter a valid concept ID."

    concept_id = int(query)
    concept = concepts.get(concept_id)
    if not concept:
        return f"No concept found with ID '{concept_id}'."

    evaluations = concept.evaluation
    if not evaluations:
        return f"No evaluations found for concept ID '{concept_id}'."

    evaluations_formatted = f"**Evaluations for {concept.label}:**\n\n"
    for eval_item in evaluations:
        evaluations_formatted += f"**{eval_item.label} ({eval_item.type.capitalize()}):**\n"
        if eval_item.type == 'multi':
            evaluations_formatted += f"Question: {eval_item.value.get('question', '')}\n"
            evaluations_formatted += f"Instruction: {eval_item.value.get('instruction', '')}\n\n"
        elif eval_item.type == 'problem':
            evaluations_formatted += f"Question: {eval_item.value.get('question', '')}\n"
            evaluations_formatted += f"Reference: {eval_item.value.get('reference', '')}\n\n"
        elif eval_item.type == 'open':
            evaluations_formatted += f"Question: {eval_item.value.get('question', '')}\n"
            evaluations_formatted += f"Instruction: {eval_item.value.get('instruction', '')}\n\n"
        else:
            evaluations_formatted += f"Details: {eval_item.value}\n\n"
    return evaluations_formatted

@tool
def update_concept_status(concept_id: str) -> str:
    """
    Update the status of a concept to 'mastery'.

    Args:
        concept_id (str): The ID number of the concept to update (e.g., '1').

    Returns:
        str: Confirmation message or error message.
    """
    concept_id = concept_id.strip()
    kst = load_kst_from_file(FILE_PATH)
    concepts = kst.concepts

    if not concept_id.isdigit():
        return f"Invalid concept ID format '{concept_id}'. Please enter a numeric concept ID (e.g., '1'). Retrieve relevant sections to see the full list of IDs if needed."

    concept_id = int(concept_id)
    concept = concepts.get(concept_id)
    if concept:
        previous_status = concept.status
        concept.status = 'mastery'
        # Also update all associated evaluations to 'complete'
        for eval_item in concept.evaluation:
            if eval_item.status.lower() != 'complete':
                eval_item.status = 'complete'
                logger.info(f"Updated status of evaluation '{eval_item.label}' to 'complete'.")
        save_kst_to_file(FILE_PATH, kst)
        logger.info(f"Updated status of concept ID {concept_id} from '{previous_status}' to 'mastery'.")
        return f"Concept ID {concept_id} ('{concept.label}') status updated to 'mastery'. All associated evaluations marked as 'complete'."
    else:
        return f"No concept found with ID '{concept_id}'."


@tool
def update_evaluation_status(evaluation_label: str) -> str:
    """
    Update the status of an evaluation to 'complete'.

    Args:
        evaluation_label (str): The label of the evaluation to update (e.g., 'E.1.1').

    Returns:
        str: Confirmation message or error message.
    """
    evaluation_label = evaluation_label.strip()
    kst = load_kst_from_file(FILE_PATH)
    concepts = kst.concepts

    if not re.fullmatch(r'E\.\d+\.\d+', evaluation_label, re.IGNORECASE):
        return "Invalid evaluation label format. Use 'E.{id}.{number}' (e.g., 'E.1.1')."

    match = re.fullmatch(r'E\.(\d+)\.(\d+)', evaluation_label, re.IGNORECASE)
    if not match:
        return "Invalid evaluation label format. Use 'E.{id}.{number}' (e.g., 'E.1.1')."

    concept_id = int(match.group(1))
    # eval_number = int(match.group(2))  # Not used currently

    concept = concepts.get(concept_id)
    if not concept:
        return f"No concept found with ID '{concept_id}'."

    # Find the evaluation by label
    eval_item = next((ev for ev in concept.evaluation if ev.label.lower() == evaluation_label.lower()), None)
    if eval_item:
        previous_status = eval_item.status
        eval_item.status = 'complete'
        # Check if all evaluations for the concept are complete
        all_complete = all(ev.status.lower() == 'complete' for ev in concept.evaluation)
        if all_complete:
            concept.status = 'mastery'
            logger.info(f"All evaluations for concept ID {concept_id} are complete. Updated concept status to 'mastery'.")
            # st.sidebar.success(f"All evaluations for concept ID {concept_id} are complete. Concept status updated to 'mastery'.")
        save_kst_to_file(FILE_PATH, kst)
        logger.info(f"Updated status of evaluation '{evaluation_label}' from '{previous_status}' to 'complete'.")
        return f"Evaluation '{evaluation_label}' for concept ID {concept_id} ('{concept.label}') status updated to 'complete'."
    else:
        return f"No evaluation found with label '{evaluation_label}' in concept ID '{concept_id}'."

@tool
def concept_connections(concept_id: str) -> Dict:
    """
    Generate a concise report of a concept's connections.

    Args:
        concept_id (str): The ID of the concept to analyze.

    Returns:
        Dict: A dictionary containing concept details and connections.
    """
    concept_id = concept_id.strip()
    if not concept_id.isdigit():
        return {"error": f"Invalid concept ID format '{concept_id}'. Please enter a numeric concept ID (e.g., '1')."}

    concept_id = int(concept_id)
    kst = load_kst_from_file(FILE_PATH)
    concepts = kst.concepts
    connections = kst.connections

    if concept_id not in concepts:
        return {"error": f"Concept ID {concept_id} does not exist in the KST."}

    current_concept = concepts[concept_id]

    # Incoming Connections: Prerequisites
    incoming = [conn['from'] for conn in connections if conn['to'] == concept_id]
    incoming_details = [
        {"id": cid, "label": concepts[cid].label, "description": concepts[cid].description}
        for cid in incoming if cid in concepts
    ]

    # Outgoing Connections: Dependents
    outgoing = [conn['to'] for conn in connections if conn['from'] == concept_id]
    outgoing_details = [
        {"id": cid, "label": concepts[cid].label, "description": concepts[cid].description}
        for cid in outgoing if cid in concepts
    ]

    # Constructing the Report
    report = {
        "concept_id": concept_id,
        "concept_label": current_concept.label,
        "description": current_concept.description,
        "incoming_connections": incoming_details,
        "outgoing_connections": outgoing_details
    }

    return report

# Define the list of tools
tools = [retrieve_concepts, retrieve_evaluations, update_concept_status, update_evaluation_status, concept_connections, search_recall_memories, save_recall_memory]

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    ks: Annotated[List[ToolMessage], operator.add]
    knowledge: str
    recall_memories: List[str]

# Define the Agent Class
class IKSOne:
    def __init__(self, model, tools, checkpointer, system):
        self.system = system

        graph = StateGraph(AgentState)
        
        graph.add_node("load_memories", self.load_memories)
        graph.add_node("knowledge_state", self.knowledge_state)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("load_memories", "knowledge_state")
        graph.add_edge("knowledge_state", "llm")
        graph.add_edge("action", "llm")
        graph.set_entry_point("load_memories")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def load_memories(self, state: AgentState):
        """Load memories for the current conversation.

        Args:
            state (schemas.State): The current state of the conversation.
            config (RunnableConfig): The runtime configuration for the agent.

        Returns:
            State: The updated state with loaded memories.
        """
        sample = filter_messages(state["messages"])
        convo_str = get_buffer_string(sample)
        convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
        recall_memories = search_recall_memories.invoke(convo_str, thread)
        return {"recall_memories": recall_memories}

    def knowledge_state(self, state: AgentState):
        """
        Generates a text representation for concepts with their statuses.
        """
        kst = load_kst_from_file(FILE_PATH)
        sorted_concepts = sorted(kst.concepts.values(), key=lambda c: c.id)
        data = {
            'Concept': [concept.label.split(' ', 1)[-1] for concept in sorted_concepts],
            'Status': [concept.status.strip() for concept in sorted_concepts]
        }
        indices = [concept.label.split(' ', 1)[0] for concept in sorted_concepts]
        df = pd.DataFrame(data, index=indices)
        
        # Exclude rows where Status is '-'
        df = df[df['Status'] != '-']
        
        return {"knowledge": df.to_string()}


    def call_llm(self, state: AgentState):
        # messages = state['messages']
        messages = trim_messages(state['messages'], strategy="last",token_counter=len,max_tokens=15,start_on="human",end_on=("human", "tool"),include_system=True)
        logger.info(messages)
        if self.system:
            messages = prompt.invoke({"messages": messages, "knowledge_state": state["knowledge"], "recall_memories": state["recall_memories"]})
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return bool(result.tool_calls)
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            tool = self.tools.get(t['name'])
            if tool:
                try:
                    # Handle tools with different arguments
                    if t['name'] == 'update_concept_status':
                        concept_id = t['args'].get('concept_id', '')
                        result = tool(concept_id)
                    elif t['name'] == 'update_evaluation_status':
                        evaluation_label = t['args'].get('evaluation_label', '')
                        result = tool(evaluation_label)
                    elif t['name'] == 'concept_connections':
                        concept_id = t['args'].get('concept_id', '0')
                        result = tool(concept_id)
                    elif t['name'] == 'save_recall_memory':
                        memory = t['args'].get('memory', '')
                        result = tool(memory)
                    else:
                        # Assume other tools take a single 'query' argument
                        result = tool(t['args'].get('query', ''))
                    results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
                except Exception as e:
                    results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=f"Error: {str(e)}"))

        state['ks'] = results

        return {'messages': results, 'ks': state['ks']}

# Initialize the LLM and Bind Tools
llm = ChatOpenAI(api_key=proxy_api,base_url="https://api.proxyapi.ru/openai/v1",model='gpt-4o',temperature=0.2).bind_tools(tools)

# Initialize Memory Saver
memory = MemorySaver()

# Initialize the Agent
ks_retriever = IKSOne(model=llm, tools=tools, system=prompt, checkpointer=memory)

# Initialize Streamlit Session State for the Agent
if 'abot' not in st.session_state:
    st.session_state.abot = ks_retriever

if 'embed' not in st.session_state:
    st.session_state.embed = recall_vector_store

if 'thread' not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": "1"}}

abot = st.session_state.abot
thread = st.session_state.thread

# Configure Streamlit Page
st.set_page_config(initial_sidebar_state="expanded")

# Custom CSS for the color theme
custom_css = """
<style>
    :root {
        --primary-color: #2B2B2B;
        --background-color: #FFFFFF;
        --text-color: #434343;
        --secondary-background-color: #F7F7F7;
        --font-family: 'Serif';
    }
</style>
"""

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Hide the Sidebar Navigation Separator
st.markdown("""<style>
    [data-testid="stSidebarNavSeparator"] {
        display: none;
    }</style>""", unsafe_allow_html=True)

# Define the Async Function to Get Responses
async def get_response(messages, thread):
    result = abot.graph.invoke({"messages": [("user", messages)]}, config=thread)
    message = result['messages'][-1]
    return message.content

# Initialize Chat History and Display Messages from History on App Rerun
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "Agent",
            "content": (
                "I'm your **assistant**. "
                "My role is to help you understand various statistical concepts, evaluate your knowledge, "
                "and track your progress as you learn.\n\n"
                "To tailor the learning experience to your needs, could you please **introduce yourself** and share your preferences and **learning style**?\n\n"
                "Here's how I can assist you:\n\n"
                "- **Learn Concepts**: I can explain statistical concepts in detail, helping you grasp key ideas step by step.\n"
                "- **Practice and Evaluate**: Once you're comfortable, I can provide practice problems and evaluations to test your understanding.\n"
                "- **Track Your Progress**: I'll automatically track your learning progress and adjust your status to reflect mastery as you improve.\n"
                "- **Customized Feedback**: Based on your performance, I'll offer targeted feedback to reinforce learning and guide you to mastery.\n\n"
                "Feel free to ask me anything to get started on your **learning journey**!"
            )
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Labels in Session State
if "labels" not in st.session_state:
    st.session_state.labels = set()

# Accept User Input
if user_input := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "Student", "content": user_input})
    
    # Display User Message
    with st.chat_message(name="Student", avatar=None):
        st.markdown(user_input)
    
    # Get current 'ks' length before invoking to track new tool calls
    if "ks" in abot.graph.get_state(thread).values:
        previous_ks_length = len(abot.graph.get_state(thread).values["ks"])
    else:
        previous_ks_length = 0  
    
    # Get Response from Assistant
    with st.chat_message(name="Agent", avatar=None):
        response = asyncio.run(get_response(user_input, thread))
        st.markdown(response)
    
    st.session_state.messages.append({"role": "Agent", "content": response})
    
    # Retrieve the updated 'ks' state after invoking
    current_ks = abot.graph.get_state(thread).values["ks"]
    # Identify only the new tool calls added in this interaction
    new_ks = current_ks[previous_ks_length:]
    
    # Reset the labels set before adding new labels
    st.session_state.labels = set()
    
    # Check and display notifications for newly invoked tools
    if any(message.name == "retrieve_evaluations" for message in new_ks):
        st.sidebar.warning("Evaluation retrieved")
    
    if any(message.name == "update_concept_status" for message in new_ks):
        st.sidebar.success("Concept status updated")
    
    if any(message.name == "update_evaluation_status" for message in new_ks):
        st.sidebar.success("Evaluation status updated")
    
    if any(message.name == "concept_connections" for message in new_ks):
        st.sidebar.warning("Concept connections retrieved")
    
    # **Added Notifications for Memory Tools**
    if any(message.name == "save_recall_memory" for message in new_ks):
        st.sidebar.success("Memory stored")
    
    if any(message.name == "search_recall_memories" for message in new_ks):
        st.sidebar.success("Memory retrieved")
    
    # Extract labels from the new ToolMessages and add to session state labels
    for message in new_ks:
        content = message.content
        # Check if the message is from an ID-based query
        if isinstance(content, str) and content.startswith("**ID_QUERY:**"):
            # Extract the label using regex
            match = re.search(r'\*\*Label:\*\* ([\w\.\s]+)', content)
            if match:
                label = match.group(1).strip()
                st.session_state.labels.add(label)

# Display labels from session state in the sidebar
if st.session_state.labels:
    for label in st.session_state.labels:
        st.sidebar.info(label)