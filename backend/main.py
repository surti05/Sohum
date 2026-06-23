from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import os
import numpy as np
import pdfplumber
import pandas as pd
import json
import io

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── In-memory stores ──────────────────────────────────────────
document_store = []
po_data = pd.read_csv("data/purchase_orders.csv")


# ── Pydantic model for /ask endpoint ─────────────────────────
class Question(BaseModel):
    question: str


# ── LangGraph State — the shared whiteboard ───────────────────
class SohumState(TypedDict):
    question: str
    needs_policy: bool
    needs_data: bool
    policy_context: str
    data_context: str
    sources: List[dict]
    used_live_data: bool
    final_answer: str


# ── Helper: embeddings ────────────────────────────────────────
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_documents(query, top_k=4):
    if not document_store:
        return []
    query_embedding = get_embedding(query)
    scored = []
    for item in document_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ── Helper: pandas tool ───────────────────────────────────────
def query_purchase_orders(pandas_query: str):
    try:
        result = po_data.query(pandas_query)
        if len(result) == 0:
            return "No matching purchase orders found."
        total_amount = result["amount_inr"].sum()
        avg_amount = result["amount_inr"].mean()
        result_dict = result.head(20).to_dict(orient="records")
        summary = {
            "matching_records_count": len(result),
            "total_amount_inr": int(total_amount),
            "average_amount_inr": round(float(avg_amount), 2),
            "records": result_dict
        }
        if len(result) > 20:
            summary["note"] = "Showing first 20 records only."
        return json.dumps(summary, default=str)
    except Exception as e:
        return f"Query error: {str(e)}. Columns: {list(po_data.columns)}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_purchase_orders",
            "description": (
                
    "Query the live purchase orders dataset. "
    "Columns: po_number, vendor, category, amount_inr, status, "
    "requestor, department, order_date, expected_delivery_date. "
    "IMPORTANT — exact status values are: 'Pending Approval', "
    "'Approved', 'Delivered', 'Rejected'. "
    "Always use these exact values with correct capitalisation. "
    "Use pandas query syntax. The result includes pre-calculated "
    "total_amount_inr and average_amount_inr — always use these "
    "pre-calculated values rather than summing yourself."
),
            
            "parameters": {
                "type": "object",
                "properties": {
                    "pandas_query": {
                        "type": "string",
                        "description": "A pandas DataFrame.query() expression"
                    }
                },
                "required": ["pandas_query"]
            }
        }
    }
]


# ══════════════════════════════════════════════════════════════
# THE FOUR AGENTS
# ══════════════════════════════════════════════════════════════

# ── Agent 1: Router ───────────────────────────────────────────
# Reads the question. Decides which modules are needed.
# Writes needs_policy and needs_data onto the whiteboard.
def router_node(state: SohumState) -> SohumState:
    question = state["question"]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a routing agent for an enterprise AI system. "
                    "Given a user question, decide which data sources are needed "
                    "to answer it. Respond with ONLY valid JSON in this exact format:\n"
                    '{"needs_policy": true/false, "needs_data": true/false}\n\n'
                    "needs_policy = true if the question is about company policies, "
                    "IT standards, SOPs, compliance rules, approval processes, "
                    "or any document-based knowledge.\n"
                    "needs_data = true if the question is about purchase orders, "
                    "vendors, spending amounts, order status, or any live data.\n"
                    "Both can be true if the question needs both sources."
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    try:
        decision = json.loads(raw)
        needs_policy = bool(decision.get("needs_policy", False))
        needs_data = bool(decision.get("needs_data", False))
    except Exception:
        needs_policy = True
        needs_data = True

    return {
        **state,
        "needs_policy": needs_policy,
        "needs_data": needs_data,
        "policy_context": "",
        "data_context": "",
        "sources": [],
        "used_live_data": False,
        "final_answer": ""
    }


# ── Agent 2: Policy Agent ─────────────────────────────────────
# Only runs if needs_policy is True.
# Searches uploaded documents and writes policy_context
# and sources onto the whiteboard.
def policy_agent_node(state: SohumState) -> SohumState:
    if not state["needs_policy"]:
        return state

    question = state["question"]
    results = search_documents(question)

    policy_context = ""
    sources = list(state.get("sources", []))

    if results:
        for score, item in results:
            policy_context += f"\n---\nSource: {item['source']}\n{item['text']}\n"
            sources.append({
                "source": item["source"],
                "relevance": round(float(score), 3)
            })

    return {
        **state,
        "policy_context": policy_context,
        "sources": sources
    }


# ── Agent 3: Data Agent ───────────────────────────────────────
# Only runs if needs_data is True.
# Queries live purchase order data using tool calling
# and writes data_context onto the whiteboard.
def data_agent_node(state: SohumState) -> SohumState:
    if not state["needs_data"]:
        return state

    question = state["question"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data retrieval agent. Your ONLY job is to query "
                "the purchase orders dataset using the available tool. "
                "Do not write a final answer — just retrieve the data."
            )
        },
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS
    )

    data_context = ""
    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"[Data Agent] Pandas query: {args['pandas_query']}")
            tool_result = query_purchase_orders(args["pandas_query"])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        data_context = tool_result
    else:
        data_context = response_message.content or ""

    return {
        **state,
        "data_context": data_context,
        "used_live_data": bool(response_message.tool_calls)
    }


# ── Agent 4: Analyst ──────────────────────────────────────────
# Always runs last.
# Reads policy_context and data_context from the whiteboard.
# Combines them and writes the final answer.
def analyst_node(state: SohumState) -> SohumState:
    question = state["question"]
    policy_context = state.get("policy_context", "")
    data_context = state.get("data_context", "")

    context_parts = []
    if policy_context:
        context_parts.append(f"POLICY / DOCUMENT CONTEXT:\n{policy_context}")
    if data_context:
        context_parts.append(f"LIVE DATA CONTEXT:\n{data_context}")

    if context_parts:
        user_message = (
            "\n\n".join(context_parts) +
            f"\n\nQuestion: {question}\n\n"
            "Answer the question using the context above. "
            "If both policy and data context are available, combine them "
            "into one coherent answer. Always cite your sources. "
            "If confidence is low, say so clearly."
        )
    else:
        user_message = question

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Sohum, an enterprise AI intelligence assistant. "
                    "Give clear, concise, business-focused answers. "
                    "Always cite which source the information came from. "
                    "If you are not sure, say so — never guess."
                )
            },
            {"role": "user", "content": user_message}
        ]
    )

    final_answer = response.choices[0].message.content
    return {**state, "final_answer": final_answer}


# ══════════════════════════════════════════════════════════════
# BUILD THE GRAPH — wire the four agents together
# ══════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(SohumState)

    graph.add_node("router", router_node)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("router")

    graph.add_edge("router", "policy_agent")
    graph.add_edge("policy_agent", "data_agent")
    graph.add_edge("data_agent", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()


sohum_graph = build_graph()


# ══════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/ask")
async def ask(q: Question):
    initial_state: SohumState = {
        "question": q.question,
        "needs_policy": False,
        "needs_data": False,
        "policy_context": "",
        "data_context": "",
        "sources": [],
        "used_live_data": False,
        "final_answer": ""
    }

    result = sohum_graph.invoke(initial_state)

    confidence = "high" if (result["policy_context"] or result["used_live_data"]) else "general knowledge"

    return {
        "question": q.question,
        "answer": result["final_answer"],
        "model": "gpt-4o",
        "confidence": confidence,
        "sources": result["sources"],
        "used_live_data": result["used_live_data"],
        "routing": {
            "needs_policy": result["needs_policy"],
            "needs_data": result["needs_data"]
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    def extract_text_with_tables(pdf_bytes):
        full_text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    headers = table[0]
                    for row in table[1:]:
                        row_text = " | ".join(
                            f"{h.strip() if h else ''}: {c.strip() if c else ''}"
                            for h, c in zip(headers, row)
                            if h and c
                        )
                        if row_text:
                            table_texts.append(row_text)
                page_text = page.extract_text() or ""
                full_text_parts.append(page_text)
                if table_texts:
                    full_text_parts.append("\n[TABLE DATA]\n" + "\n".join(table_texts))
        return "\n\n".join(full_text_parts)

    def chunk_text(text, chunk_size=600, overlap=100):
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
            while len(current_chunk) > chunk_size * 2:
                chunks.append(current_chunk[:chunk_size].strip())
                current_chunk = current_chunk[chunk_size - overlap:]
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    full_text = extract_text_with_tables(contents)
    chunks = chunk_text(full_text)

    for chunk in chunks:
        embedding = get_embedding(chunk)
        document_store.append({
            "text": chunk,
            "embedding": embedding,
            "source": file.filename
        })

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "total_chunks_in_store": len(document_store)
    }


@app.get("/health")
async def health():
    return {
        "status": "Sohum is running — Phase 2 (multi-agent)",
        "documents_indexed": len(document_store),
        "architecture": "LangGraph — Router → Policy Agent → Data Agent → Analyst"
    }