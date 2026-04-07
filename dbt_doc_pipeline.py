import os
import sys
import json
import logging
import re
from typing import TypedDict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import config_path
from custom_slm import slm
from langgraph.graph import StateGraph, END

# -------------------- CONFIG --------------------
with open(os.path.join(config_path, "config.json"), "r", encoding="utf-8") as CONFIG:
    config = json.load(CONFIG)

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(base_dir, "qdrant_dbt_schemes_storage")
COLLECTION_NAME = "dbt_scheme_embeddings"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path=QDRANT_PATH)

# -------------------- FILTER + RETRIEVAL --------------------

def retrieve_scheme_names(question: str):
    records, _ = client.scroll(collection_name=COLLECTION_NAME, limit=2000)
    filters = extract_filters_slm(question)
    #logger.info(f"[FILTERS] Extracted: {filters}")
    logger.info(f"[FILTERS AFTER NORMALIZATION]: {filters}")

    matched_schemes = []

    for r in records:
        payload = r.payload or {}
        scheme_name = payload.get("scheme_name", "")
        summary = payload.get("summary", "")

        if not scheme_name:
            continue

        if match_filters(payload, filters):
            matched_schemes.append({
                "scheme_name": scheme_name,
                "summary": summary
            })

    logger.info(f"Matched Schemes are: {matched_schemes}")
    return matched_schemes


def retrieve_relevant_chunks(question, k=5):
    query_vector = embedding_model.encode(question).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        scheme_name = payload.get("scheme_name", "")
        eligibility = payload.get("eligibility", [])
        benefits = payload.get("benefits", [])
        documents = payload.get("documents", [])
        summary = payload.get("summary", "")

        eligibility_text = "\n".join([f"- {e}" for e in eligibility])
        benefits_text = "\n".join([
            "- " + ", ".join(f"{k}: {v}" for k, v in b.items())
            for b in benefits
        ])
        documents_text = "\n".join([
            "- " + ", ".join(f"{k}: {v}" for k, v in d.items())
            for d in documents
        ])

        results.append(f"""
Scheme: {scheme_name}

Summary:
{summary}

Eligibility:
{eligibility_text}

Benefits:
{benefits_text}

Documents:
{documents_text}
""".strip())

    return results

# -------------------- FORMAT --------------------

def format_scheme_list(question: str, schemes):
    if not schemes:
        return "No schemes found for given criteria."

    answer = "Schemes with specified criteria:\n\n"

    for s in schemes:
        answer += f"- {s['scheme_name']}\n"

    return answer.strip()


def format_count_answer(question: str, count: int) -> str:
    if count == 0:
        return "There are no schemes available for your criteria."

    prompt = f"""
Facts:
Count = {count}

Rules:
- One simple sentence
- Very easy English

Question:
{question}

Answer:
"""
    return slm.invoke(prompt).strip()


def normalize_classes(class_list):
    normalized = []

    for c in class_list:
        if isinstance(c, int):
            normalized.append(c)
        elif isinstance(c, str):
            # remove non-digits (7th -> 7)
            num = re.sub(r"\D", "", c)
            if num:
                normalized.append(int(num))

    return normalized

# -------------------- FILTER LOGIC --------------------

def extract_filters_slm(question: str):
    prompt = f"""
Extract filters as JSON.

Question: {question}

Output format:
{{
  "caste": [],
  "gender": null,
  "classes": [],
  "class_type": null,
  "hosteller": null,
  "disabled": null
}}
"""
    response = slm.invoke(prompt)

    try:
        #return json.loads(response.strip())
        data = json.loads(response.strip())
        data["classes"] = normalize_classes(data.get("classes", []))
        return data
    except:
        match = re.search(r"\{{.*\}}", response, re.DOTALL)
        if match:
            #return json.loads(match.group())
            data = json.loads(match.group())
            data["classes"] = normalize_classes(data.get("classes", []))
            return data

    return {
        "caste": [],
        "gender": None,
        "classes": [],
        "class_type": None,
        "hosteller": None,
        "disabled": None
    }


def match_filters(payload, filters):
    metadata = payload.get("metadata", {})

    # Normalize metadata
    eligible_castes = metadata.get("Eligible Castes", [])
    eligible_genders = metadata.get("Eligible Genders", [])
    eligible_classes = metadata.get("Eligible Classes", [])
    hostellers = metadata.get("Applies To Hostellers", [])
    disabled = metadata.get("Applies To Disabled", [])

    # ---------- CASTE ----------
    if filters.get("caste"):
        # filters["caste"] is a list
        if not any(
            any(fc.lower() in ec.lower() for ec in eligible_castes)
            for fc in filters["caste"]
        ):
            return False

    # ---------- GENDER ----------
    if filters.get("gender"):
        if not any(filters["gender"].lower() in g.lower() for g in eligible_genders):
            return False

    # ---------- CLASS TYPE ----------
    if filters.get("class_type"):
        if filters["class_type"] == "pre_matric":
            if not any(c <= 10 for c in eligible_classes):
                return False

        elif filters["class_type"] == "post_matric":
            if not any(c >= 11 for c in eligible_classes):
                return False

    # ---------- EXACT CLASSES ----------
    if filters.get("classes"):
        # check overlap between requested classes and scheme classes
        if not any(c in eligible_classes for c in filters["classes"]):
            return False

    # ---------- HOSTELLER ----------
    if filters.get("hosteller") is True:
        # empty list means not applicable
        if not hostellers:
            return False

    # ---------- DISABLED ----------
    if filters.get("disabled") is True:
        if not disabled:
            return False

    return True

# -------------------- INTENT --------------------

def intent_classification(question: str):
    prompt = f"""
Classify intent:

- list_schemes
- count_schemes
- scheme_details

Question: {question}

Output JSON only.
"""
    text = slm.invoke(prompt)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    return {"intent": "scheme_details"}

# ================= LANGGRAPH =================

class GraphState(TypedDict):
    question: str
    intent: str
    schemes: list
    count: int
    chunks: list
    answer: str

# -------------------- NODES --------------------

def intent_node(state: GraphState):
    res = intent_classification(state["question"])
    state["intent"] = res.get("intent", "scheme_details")
    return state


def list_or_count_node(state: GraphState):
    schemes = retrieve_scheme_names(state["question"])
    state["schemes"] = schemes
    state["count"] = len(schemes)
    return state


def format_list_node(state: GraphState):
    state["answer"] = format_scheme_list(state["question"], state["schemes"])
    return state


def format_count_node(state: GraphState):
    state["answer"] = format_count_answer(state["question"], state["count"])
    return state


def retrieve_chunks_node(state: GraphState):
    state["chunks"] = retrieve_relevant_chunks(state["question"])
    return state


def generate_answer_node(state: GraphState):
    if not state["chunks"]:
        state["answer"] = "can't answer this question based on available documents"
        return state

    context = "\n\n".join(state["chunks"])

    prompt = f"""
Answer using context only.

Context:
{context}

Question:
{state["question"]}
"""
    response = slm.invoke(prompt).strip()
    state["answer"] = re.sub(r"<think>.*?</think>", "", response)
    return state

# -------------------- ROUTING --------------------

def route_intent(state: GraphState):
    if state["intent"] in ["list_schemes", "count_schemes"]:
        return "list_or_count"
    return "retrieve_chunks"


def route_list_or_count(state: GraphState):
    if state["intent"] == "count_schemes":
        return "format_count"
    return "format_list"

# -------------------- BUILD GRAPH --------------------

builder = StateGraph(GraphState)

builder.add_node("intent", intent_node)
builder.add_node("list_or_count", list_or_count_node)
builder.add_node("format_list", format_list_node)
builder.add_node("format_count", format_count_node)
builder.add_node("retrieve_chunks", retrieve_chunks_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("intent")

builder.add_conditional_edges("intent", route_intent, {
    "list_or_count": "list_or_count",
    "retrieve_chunks": "retrieve_chunks"
})

builder.add_conditional_edges("list_or_count", route_list_or_count, {
    "format_list": "format_list",
    "format_count": "format_count"
})

builder.add_edge("retrieve_chunks", "generate_answer")

builder.add_edge("format_list", END)
builder.add_edge("format_count", END)
builder.add_edge("generate_answer", END)

graph = builder.compile()

# -------------------- WRAPPER --------------------

def generate_answer_from_documents(question: str):
    result = graph.invoke({"question": question})
    return result.get("answer", "")

# -------------------- MAIN --------------------

if __name__ == "__main__":
    q = "i am studying 7th standard belongs to sc category, what schemes am i eligible for?"
    print(f"Q: {q}")
    print("A: ", generate_answer_from_documents(q))
