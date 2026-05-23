# STEP 0: Install dependencies (run once)
# !pip install -qU langgraph langchain langchain-google-genai google-generativeai

# ---------------- IMPORTS ----------------
import os
import getpass
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------- API KEY ----------------
os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter Gemini API Key: ")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.3
)


# ---------------- STATE ----------------
class State(TypedDict):
    symptom: str
    category: str
    answer: str


# ---------------- STEP 1: INPUT ----------------
def get_symptom(state: State) -> State:
    symptom = input("Welcome to hospital, please enter your symptoms: ")
    state["symptom"] = symptom
    return state


# ---------------- STEP 2: CLASSIFICATION ----------------
def classify_symptom(state: State) -> State:
    symptom = state["symptom"].lower()

    if any(word in symptom for word in ["accident", "bleeding", "broken", "fracture"]):
        category = "emergency"
    elif any(word in symptom for word in ["sad", "depressed", "anxiety", "stress"]):
        category = "mental_health"
    else:
        category = "general"

    print(f"LLM classifies the symptom as: {category.capitalize()}")
    state["category"] = category
    return state


# ---------------- STEP 3: ROUTER ----------------
def symptom_router(state: State):
    return state["category"]


# ---------------- STEP 4: NODES ----------------
def general_node(state: State) -> State:
    state["answer"] = f"{state['symptom']}: seems general - directing to general ward"
    return state


def emergency_node(state: State) -> State:
    state["answer"] = f"{state['symptom']}: seems emergency - directing to emergency ward"
    return state


def mental_health_node(state: State) -> State:
    state["answer"] = f"{state['symptom']}: seems mental health - directing to mental health ward"
    return state


# ---------------- STEP 5: LLM FEEDBACK ----------------
def llm_feedback(state: State) -> State:
    prompt = f"""
You are a helpful hospital assistant.

Symptom: {state['symptom']}
Category: {state['category']}
Decision: {state['answer']}

Give short helpful feedback to patient.
"""

    response = llm.invoke(prompt)
    state["answer"] += "\n\nAI Feedback: " + response.content
    return state


# ---------------- STEP 6: BUILD GRAPH ----------------
builder = StateGraph(State)

builder.set_entry_point("get_symptom")

builder.add_node("get_symptom", get_symptom)
builder.add_node("classify", classify_symptom)
builder.add_node("general", general_node)
builder.add_node("emergency", emergency_node)
builder.add_node("mental_health", mental_health_node)
builder.add_node("feedback", llm_feedback)

builder.add_edge("get_symptom", "classify")

builder.add_conditional_edges(
    "classify",
    symptom_router,
    {
        "general": "general",
        "emergency": "emergency",
        "mental_health": "mental_health",
    }
)

builder.add_edge("general", "feedback")
builder.add_edge("emergency", "feedback")
builder.add_edge("mental_health", "feedback")

builder.add_edge("feedback", END)


# ---------------- STEP 7: RUN ----------------
graph = builder.compile()

final_state = graph.invoke({})

print("\nFinal Output:\n")
print(final_state["answer"])