import os 
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import redis
import hashlib
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv


load_dotenv()

print("Gemini Key:", os.getenv("GEMINI_API_KEY"))


llm =ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                            google_api_key=os.getenv("GEMINI_API_KEY"),
                            temperature=0.7)

# simple prompt 
response =llm.invoke("Suggest a 3-day itinerary for Tokyo focused on anime, sushi, and temples.")
print(response.content)


def generate_travel_prompt(user_query, memories=None):

    if memories is None:
        memories = []

    memory_context = "\n".join(memories)

    prompt = f"""
    You are an intelligent AI Travel Assistant.

    User Preferences:
    {memory_context}

    User Query:
    {user_query}

    Provide:
    - destination highlights
    - local food recommendations
    - cultural experiences
    - travel tips
    - itinerary suggestions

    Generate a detailed and personalized travel response.
    """

    return prompt



def generate_fingerprint(text):
    normalised_text = text.strip().lower()
    fingerprint = hashlib.sha256(normalised_text.encode()).hexdigest()
    return fingerprint

print(generate_fingerprint("Suggest a honeymoon trip to Bali") )
query1 = "I want to visit Tokyo"

query2 = "i want to visit tokyo"

fp1 = generate_fingerprint(query1)
fp2 = generate_fingerprint(query2)

print(fp1)
print(fp2)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
redis_client.ping()

def store_in_cache(fingerprint, response_data):
    redis_client.set(
        fingerprint,
        json.dumps(response_data)
    )

def get_from_cache(fingerprint):
    cached_data = redis_client.get(fingerprint)
    if cached_data:
        return json.loads(cached_data)
    return None


#manual redis test
sample_data={"response": "Visit bali for vacation"}
store_in_cache("test_fingerprint", sample_data)
cached = get_from_cache("test_fingerprint")
print(cached)



embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)
semantic_cache=[]
def generate_embedding(text):
    embedding= embedding_model.encode(text)
    return embedding

def find_similar_query(query, threshold=0.85):

    query_embedding = generate_embedding(query)

    best_match = None
    best_score = 0

    for item in semantic_cache:

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        if similarity > best_score:

            best_score = similarity
            best_match = item

    if best_score >= threshold:

        return best_match

    return None


query1 = "I want beach vacations"

query2 = "Suggest tropical destinations"

emb1 = generate_embedding(query1)
emb2 = generate_embedding(query2)

score = cosine_similarity(
    [emb1],
    [emb2]
)[0][0]

print(score)

def ask_travel_assistant(query):

    similar_match = find_similar_query(query)

    if similar_match:

        cached_response = get_from_cache(
            similar_match["fingerprint"]
        )

        if cached_response:

            cached_response["source"] = "semantic_cache"

            return cached_response

    fingerprint = generate_fingerprint(query)

    prompt = generate_travel_prompt(query)

    start_time = time.time()

    response = llm.invoke(prompt)

    latency = (time.time() - start_time) * 1000

    result = {
        "query": query,
        "fingerprint": fingerprint,
        "response": response.content,
        "latency_ms": latency,
        "source": "llm"
    }

    store_in_cache(fingerprint, result)

    semantic_cache.append({
        "query": query,
        "embedding": generate_embedding(query),
        "fingerprint": fingerprint
    })

    return result

user_memories = []
def add_memory(user_id, memory_text):

    embedding = generate_embedding(memory_text)

    user_memories.append({
        "user_id": user_id,
        "memory": memory_text,
        "embedding": embedding
    })

def retrieve_memories(user_id, query, threshold=0.65):

    query_embedding = generate_embedding(query)

    relevant_memories = []

    for memory in user_memories:

        if memory["user_id"] != user_id:
            continue

        similarity = cosine_similarity(
            [query_embedding],
            [memory["embedding"]]
        )[0][0]

        if similarity >= threshold:

            relevant_memories.append(
                memory["memory"]
            )

    return relevant_memories

add_memory(
    "jithin",
    "User prefers cold destinations"
)

memories = retrieve_memories(
    "jithin",
    "Suggest cold places for vacation"
)

print(memories)

memories = retrieve_memories(
    "jithin",
    "Suggest cold places for vacation"
)


from typing import TypedDict
from langgraph.graph import StateGraph, END
class TravelState(TypedDict):

    query: str

    user_id: str

    memories: list

    fingerprint: str

    cached_response: dict

    response: str

    source: str

def memory_node(state):

    memories = retrieve_memories(
        state["user_id"],
        state["query"]
    )

    state["memories"] = memories

    return state
def fingerprint_node(state):

    fingerprint = generate_fingerprint(
        state["query"]
    )

    state["fingerprint"] = fingerprint

    return state
def cache_node(state):

    similar_match = find_similar_query(
        state["query"]
    )

    if similar_match:

        cached_response = get_from_cache(
            similar_match["fingerprint"]
        )

        if cached_response:

            state["cached_response"] = cached_response

            state["response"] = cached_response["response"]

            state["source"] = "semantic_cache"

    return state
def llm_node(state):

    if state.get("response"):

        return state

    prompt = generate_travel_prompt(
        state["query"],
        state["memories"]
    )

    response = llm.invoke(prompt)

    state["response"] = response.content

    state["source"] = "llm"

    return state
def store_cache_node(state):

    if state["source"] == "llm":

        result = {
            "query": state["query"],
            "response": state["response"],
            "source": state["source"]
        }

        store_in_cache(
            state["fingerprint"],
            result
        )

        semantic_cache.append({
            "query": state["query"],
            "embedding": generate_embedding(
                state["query"]
            ),
            "fingerprint": state["fingerprint"]
        })

    return state

#adding nodes
workflow = StateGraph(TravelState)
workflow.add_node(
    "memory",
    memory_node
)

workflow.add_node(
    "fingerprint",
    fingerprint_node
)

workflow.add_node(
    "cache",
    cache_node
)

workflow.add_node(
    "llm",
    llm_node
)

workflow.add_node(
    "store_cache",
    store_cache_node
)

workflow.set_entry_point("memory")

workflow.add_edge(
    "memory",
    "fingerprint"
)

workflow.add_edge(
    "fingerprint",
    "cache"
)

workflow.add_edge(
    "cache",
    "llm"
)

workflow.add_edge(
    "llm",
    "store_cache"
)

workflow.add_edge(
    "store_cache",
    END
)
travel_graph = workflow.compile()

flash_lite_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)
flash_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

def compare_models(query, memories=None):

    if memories is None:
        memories = []

    prompt = generate_travel_prompt(
        query,
        memories
    )

    # FLASH LITE
    lite_start = time.time()

    lite_response = flash_lite_llm.invoke(prompt)

    lite_latency = (
        time.time() - lite_start
    ) * 1000

    # FLASH FULL
    flash_start = time.time()

    flash_response = flash_llm.invoke(prompt)

    flash_latency = (
        time.time() - flash_start
    ) * 1000

    return {

        "query": query,

        "lite_response": lite_response.content,

        "lite_latency_ms": lite_latency,

        "flash_response": flash_response.content,

        "flash_latency_ms": flash_latency
    }

comparison = compare_models(
    "Suggest a luxury winter vacation in Switzerland"
)
print("FLASH LITE LATENCY:")
print(comparison["lite_latency_ms"])

print("\nFLASH LATENCY:")
print(comparison["flash_latency_ms"])

class TravelRequest(BaseModel):

    query: str

    user_id: str = "jithin"

app = FastAPI(
    title="Intelligent Travel Query API"
)

@app.post("/memory-travel-assistant")

def memory_travel_assistant(
    request: TravelRequest
):

    try:

        query = request.query.strip()

        if not query:

            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        result = travel_graph.invoke({

            "query": query,

            "user_id": request.user_id
        })

        memories = result.get(
            "memories",
            []
        )

        return {

            "response": result["response"],

            "source": result["source"],

            "metadata": {

                "fingerprint_hit":
                    result["source"] != "llm",

                "cache_hit":
                    result["source"] == "semantic_cache",

                "memories_retrieved":
                    memories
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
