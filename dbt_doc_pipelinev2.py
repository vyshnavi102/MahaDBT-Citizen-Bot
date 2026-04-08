import os
import sys
import json
import re
from typing import TypedDict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import config_path
from custom_slm import slm
from langgraph.graph import StateGraph, END
from config.logger_util import get_logger, LogTimer, log_execution, get_current_logger

logger = get_current_logger()

# -------------------- CONFIG --------------------
with open(os.path.join(config_path, "config.json"), "r", encoding="utf-8") as CONFIG:
    config = json.load(CONFIG)

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

base_dir = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(base_dir, "qdrant_dbt_schemes_storage")
COLLECTION_NAME = "dbt_scheme_embeddingsv2"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path=QDRANT_PATH)

# -------------------- FILTER + RETRIEVAL --------------------

def retrieve_scheme_names(question: str, logger):
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


# def retrieve_relevant_chunks(question, k=5):
#     query_vector = embedding_model.encode(question).tolist()
#     hits = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=k
#     )

#     results = []
#     for hit in hits:
#         payload = hit.payload or {}
#         scheme_name = payload.get("scheme_name", "")
#         eligibility = payload.get("eligibility", [])
#         benefits = payload.get("benefits", [])
#         documents = payload.get("documents", [])
#         summary = payload.get("summary", "")

#         eligibility_text = "\n".join([f"- {e}" for e in eligibility])
#         benefits_text = "\n".join([
#             "- " + ", ".join(f"{k}: {v}" for k, v in b.items())
#             for b in benefits
#         ])
#         documents_text = "\n".join([
#             "- " + ", ".join(f"{k}: {v}" for k, v in d.items())
#             for d in documents
#         ])

#         results.append(f"""
# Scheme: {scheme_name}

# Summary:
# {summary}

# Eligibility:
# {eligibility_text}

# Benefits:
# {benefits_text}

# Documents:
# {documents_text}
# """.strip())

#     return results

def retrieve_relevant_chunks(question, k=5):
    filters = extract_filters_slm(question)

    query_vector = embedding_model.encode(question).tolist()

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=20   # increase pool
    )

    results = []

    for hit in hits:
        payload = hit.payload or {}

        if not match_filters(payload, filters):
            continue

        scheme_name = payload.get("scheme_name", "")
        department_name = payload.get("department_name", "")
        department_description = payload.get("department_description", "")
        eligibility = payload.get("eligibility", [])
        benefits = payload.get("benefits", [])
        documents = payload.get("documents", [])
        summary = payload.get("summary", "")

        eligibility_text = "\n".join([f"- {e}" for e in eligibility])
        # benefits_text = "\n".join([
        #     "- " + ", ".join(f"{k}: {v}" for k, v in b.items())
        #     for b in benefits
        # ])
        benefits_text = "\n".join([
            "- " + ", ".join(f"{k}: {v}" for k, v in b.items() if v)
            for b in benefits
        ])
        # documents_text = "\n".join([
        #     "- " + ", ".join(f"{k}: {v}" for k, v in d.items())
        #     for d in documents
        # ])
        documents_text = "\n".join([
            f"- {d}" if isinstance(d, str)
            else "- " + ", ".join(f"{k}: {v}" for k, v in d.items())
            for d in documents
        ])

        results.append(f"""
Scheme: {scheme_name}

Department:
{department_name}

Department Description:
{department_description}

Summary:
{summary}

Eligibility:
{eligibility_text}

Benefits:
{benefits_text}

Documents:
{documents_text}
""".strip())

        if len(results) >= k:
            break

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

def detect_scope_slm(question: str):
    prompt = f"""
You are a STRICT scope classifier for a Government Student Scheme Assistant.

Your job:
Decide whether the user question can be answered using STUDENT SCHEME DATA.

-----------------------------------
IMPORTANT: WHAT EXISTS IN DATABASE

The database contains schemes related to:

- Scholarships (pre-matric, post-matric, merit)
- Maintenance allowance
- Tuition fees / exam fees / freeship
- Hosteller benefits
- Sainik school schemes
- Education financial assistance
- Caste-based student schemes (SC, OBC, VJNT, SBC)
- School students (class 1 to 10 mainly)

-----------------------------------
CLASSIFICATION RULES (VERY IMPORTANT)

Return TRUE if the question is related to ANY of the following:

- Students / school education
- Financial help for students
- Scholarships / allowance / freeship / stipend
- Eligibility / benefits / documents of schemes
- Specific scheme names
- School types (Sainik school, govt school, hostel, etc.)
- Queries like:
    "eligibility", "benefits", "documents", "how much amount"
    "which schemes", "available schemes"

EVEN IF the word "scholarship" is NOT present → STILL TRUE

-----------------------------------
Return FALSE ONLY IF:

- Question is about:
    - jobs / recruitment
    - farmers / agriculture
    - pensions
    - business / loans
    - unrelated topics

-----------------------------------
CRITICAL RULE:

If the question contains ANY of these → ALWAYS TRUE:
- student
- class / std
- school
- scholarship
- allowance
- freeship
- Sainik school
- hostel / hosteller

DO NOT be overly strict.
Prefer TRUE if unsure.

-----------------------------------

Return ONLY JSON:

{{
  "in_scope": true/false,
  "reason": "short reason"
}}

-----------------------------------

Question:
{question}
"""
    try:
        response = slm.invoke(prompt)

        import re, json
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    except Exception as e:
        print("Scope detection error:", e)

    return {"in_scope": True, "reason": "fallback"}


# -------------------- FILTER LOGIC --------------------

def extract_filters_slm(question: str):
    prompt = f"""
You are a strict JSON filter extractor.

Your task:
Extract filter values from the question.

------------------------
OUTPUT RULES (VERY STRICT):

- Return ONLY valid JSON
- No explanation
- No extra text
- Start with {{ and end with }}
- Use double quotes only
- No trailing commas

------------------------
EXTRACTION INSTRUCTIONS (VERY IMPORTANT):

1. CASTE:
- If ANY caste word is present in the question (SC, ST, OBC, SBC, VJNT, etc.)
  → YOU MUST add it to "caste"
- Do NOT skip it
- Do NOT leave it empty if present
- Return as list
- Example: "sc students" → ["SC"]

2. GENDER:
- If ANY gender-related word is present
  → YOU MUST set gender
- girls/female → "Female"
- boys/male → "Male"
- Do NOT leave null if present

3. CLASSES (VERY IMPORTANT - GENERAL RULE FOR ALL CLASSES):

- If ANY class number is mentioned in the question → YOU MUST extract it
- This applies to ALL class numbers (1 to 12), not just specific examples

You MUST handle ALL formats:

1. Numbers:
- "class 8", "class 9", "class 10" → [8], [9], [10]

2. Ordinal forms:
- "8th", "9th", "10th" → [8], [9], [10]

3. Multiple classes:
- "6th and 7th" → [6,7]
- "8, 9, 10" → [8,9,10]

4. Range:
- "6 to 8" → [6,7,8]
- "6th to 8th" → [6,7,8]

5. ROMAN NUMERALS (VERY IMPORTANT - APPLY TO ALL):

- Roman numerals represent class numbers from 1 to 12

You MUST convert ALL roman numerals correctly:

I → 1
II → 2
III → 3
IV → 4
V → 5
VI → 6
VII → 7
VIII → 8
IX → 9
X → 10
XI → 11
XII → 12

------------------------

RANGE HANDLING:

- If a range is given → convert both and include all numbers between

Examples:

- "VIII" → [8]
- "VII" → [7]
- "X" → [10]

- "VIII to X" → [8,9,10]
- "VII to IX" → [7,8,9]
- "V to VII" → [5,6,7]

------------------------

IMPORTANT:

- Apply this conversion to ALL roman numerals, not just IX
- Do NOT skip any roman numeral
- Always convert to integers
- Always return full range when "to" or "-" is present

6. Mixed:
- "8th to X" → [8,9,10]
- "IX to 12" → [9,10,11,12]

IMPORTANT:
- Apply this logic to ALL class numbers (1–12)
- Do NOT focus only on examples like 9
- Ignore words like "class", "th", "std"
- Always convert to integers
- Always return a list

- If class is present → MUST extract
- If not present → return []

4. CLASS TYPE:
- If "pre matric" → "pre_matric"
- If "post matric" → "post_matric"
- If class_type is present → classes MUST be []

5. HOSTELLER:
- If "hostel" or "hosteller" is present
  → YOU MUST set "yes"

6. DISABLED:
- If "disabled" or "handicapped" is present
  → YOU MUST set "yes"

7. HIGHER EDUCATION (VERY IMPORTANT):

If question contains ANY of:
- degree
- college
- graduation
- ug / pg
- engineering
- medical
- iti
- diploma

→ YOU MUST set:
"class_type": "post_matric"

→ AND classes MUST be []

------------------------
CRITICAL RULES:

- If a value is clearly present in the question → YOU MUST extract it
- Do NOT ignore obvious words like "sc", "girls", "class 9"
- Do NOT leave fields empty if they are present in the question
- Only leave empty/null if truly not mentioned

------------------------
OUTPUT FORMAT:

{{
  "caste": [],
  "gender": null,
  "classes": [],
  "class_type": null,
  "hosteller": null,
  "disabled": null
}}

------------------------

Question:
{question}

Output:
"""
    response = slm.invoke(prompt)

    try:
        #return json.loads(response.strip())
        data = json.loads(response.strip())
        data["classes"] = normalize_classes(data.get("classes", []))
        if data.get("classes"):
            data["class_type"] = None
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
        if not any(
            g.lower() == "all" or filters["gender"].lower() == g.lower()
            for g in eligible_genders
        ):
            return False

    # ---------- CLASS TYPE ----------
    if filters.get("class_type") and not filters.get("classes"):
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
    if filters.get("hosteller") in [True, "yes"]:
        # empty list means not applicable
        if not hostellers:
            return False

    # ---------- DISABLED ----------
    if filters.get("disabled") in [True, "yes"]:
        if not disabled:
            return False

    return True

def route_scope(state):
    if state.get("in_scope") == False:
        return "out_of_scope"
    return "intent"

# -------------------- INTENT --------------------

def intent_classification(question: str):
    prompt = f"""
You are an intent classifier.

Classify the user question into EXACTLY ONE of the following intents:

1. count_schemes
   - The user is asking for the count, number of schemes.
   - Example: "How many schemes are there?"

2. scheme_details
   - The user is asking about a specific scheme or detailed info.
   - Includes questions about:
     - eligibility
     - benefits
     - documents
     - application process
     - deadlines
     - detailed information of a particular scheme
   - If the user asks about eligibility, benefits, documents, or any detailed info → return scheme_details.

3. list_schemes
   - The user is asking for the names of multiple schemes.
   - It can include simple filters like caste, gender, class, or class type.
   - Must NOT ask for detailed information like eligibility, benefits, or documents.
   - Example: "List schemes for girls"
   - Example: "Show post matric schemes for SC students"

IMPORTANT RULES:
- If the question mentions a specific scheme name → ALWAYS return "scheme_details".
- If the question explicitly asks about eligibility, benefits, documents, or deadlines → return "scheme_details".
- If the question is general or filtered (gender, caste, class, hosteller, disabled) → return "list_schemes".
- If the question asks "how many" or "number of" → return "count_schemes".
- Return ONLY valid JSON.
- Do NOT add explanations or extra text.
- If question is general like:
  "what schemes are available"
  "what financial help"
  "any scholarships"

→ return "list_schemes"

Question:
{question}

Final Output:
{{"intent":"list_schemes"}}
{{"intent":"count_schemes"}}
{{"intent":"scheme_details"}}
"""
    text = slm.invoke(prompt)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        data = json.loads(match.group())
        if data.get("intent") in [
            "list_schemes",
            "count_schemes",
            "scheme_details"
        ]:
            return data
    return {"intent": "scheme_details"}

# ================= LANGGRAPH =================

class GraphState(TypedDict):
    question: str
    intent: str
    schemes: list
    count: int
    chunks: list
    answer: str
    in_scope: bool
    scope_reason: str


# -------------------- NODES --------------------

def intent_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "intent_node"):
        res = intent_classification(state["question"])
        state["intent"] = res.get("intent", "scheme_details")

        logger.info(f"Detected intent: {state['intent']}")

    return state


def list_or_count_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "list_or_count_node"):
        schemes = retrieve_scheme_names(state["question"], logger)

        state["schemes"] = schemes
        state["count"] = len(schemes)

        logger.info(f"Schemes found: {state['count']}")

    return state


def format_list_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "format_list_node"):
        state["answer"] = format_scheme_list(state["question"], state["schemes"])

    return state

def format_count_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "format_count_node"):
        state["answer"] = format_count_answer(state["question"], state["count"])

    return state

def retrieve_chunks_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "retrieve_chunks_node"):
        state["chunks"] = retrieve_relevant_chunks(state["question"])

        logger.info(f"Chunks retrieved: {len(state['chunks'])}")

    return state

def scope_node(state: dict):
    question = state["question"]

    result = detect_scope_slm(question)

    state["in_scope"] = result.get("in_scope", True)
    state["scope_reason"] = result.get("reason", "")

    return state

def out_of_scope_node(state: dict):
    return {
        "answer": "Currently, we provide only student scholarship schemes. Your query seems outside this scope."
    }

def generate_answer_node(state: GraphState):
    logger = get_current_logger()

    with LogTimer(logger, "generate_answer_node"):

        if not state["chunks"]:
            logger.warning("No chunks found")
            state["answer"] = "No schemes found for given criteria."
            return state

        context = "\n\n".join(state["chunks"])

        prompt = f"""
You are a government scheme assistant.

STRICT RULES:
1. Answer ONLY using the given context.
2. Do NOT assume missing information.
3. Use very simple English.
4. Keep sentences short.
5. No unnecessary explanation.

CRITICAL RULE (VERY IMPORTANT):
- Identify what the user is asking:
    • If question is about eligibility → ONLY show Eligibility
    • If question is about benefits → ONLY show Benefits
    • If question is about documents → ONLY show Documents
    • If question is about summary → ONLY show Summary
    • If question is general → show all relevant fields

- DO NOT include any other fields.
- DO NOT dump full scheme details unless explicitly asked.

FORMATTING RULES:
- Make the answer clean and readable.
- Use proper spacing.
- Highlight headings in bold.

OUTPUT FORMAT:

**<Scheme Name>**

(Only include the requested section below)

**Eligibility:**
- point 1
- point 2

OR

**Benefits:**
- point 1
- point 2

OR

**Documents:**
- point 1
- point 2

OR

**Summary:**
- short explanation


IMPORTANT:
- Do NOT include extra sections.
- Do NOT include unnecessary text.
- Keep output minimal and precise.

Context:
{context}

Question:
{state["question"]}

Answer:
"""

        response = slm.invoke(prompt).strip()
        state["answer"] = re.sub(r"<think>.*?</think>", "", response)

        logger.info("Answer generated successfully")

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

builder.add_node("scope", scope_node)
builder.add_node("intent", intent_node)
builder.add_node("out_of_scope", out_of_scope_node)
builder.add_node("list_or_count", list_or_count_node)
builder.add_node("format_list", format_list_node)
builder.add_node("format_count", format_count_node)
builder.add_node("retrieve_chunks", retrieve_chunks_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("scope")

builder.add_conditional_edges(
    "scope",
    route_scope,
    {
        "intent": "intent",
        "out_of_scope": "out_of_scope"
    }
)

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
    logger = get_logger()

    logger.info(f"Incoming question: {question}")

    with LogTimer(logger, "run_pipeline"):
        result = graph.invoke({
            "question": question
        })
    logger.info(f"Final answer: {result.get('answer')}")

    logger.info("Pipeline finished successfully")

    return result.get("answer", "")

# -------------------- MAIN --------------------

if __name__ == "__main__":
    q = "list the schemes available?"
    print(f"Q: {q}")
    print("A: ", generate_answer_from_documents(q))

