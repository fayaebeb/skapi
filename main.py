from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

# from astra_retriever import retriever
from message_formatter import format_as_message
from tavily_search import tavily_search
from google_search import google_search

from langchain_astradb import AstraDBVectorStore

load_dotenv()

app = FastAPI()

class ChatInput(BaseModel):
    message: str
    useweb: bool = False
    usedb: bool = False
    db: str = "files"
    history: Optional[List[dict]] = None
    
def convert_history_to_messages(history: Optional[List[dict]]) -> List:
    messages = []
    if history:
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    return messages

def deduplicate_docs(docs):
    unique = []
    seen = set()
    for doc in docs:
        identifier = (doc.page_content, frozenset(doc.metadata.items()) if hasattr(doc, 'metadata') else None)
        if identifier not in seen:
            unique.append(doc)
            seen.add(identifier)
    return unique


def generate_search_query(user_message: str) -> str:
    """
    Ask the LLM to rewrite the user's input into a concise
    keyword‐rich query for vector search.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
    )

    query_system = SystemMessage(
        content=(
            "You are a smart assistant that rewrites user questions "
            "into concise search queries for an internal knowledge base. "
            "Focus on keywords and core concepts."
        )
    )

    query_user = HumanMessage(
        content=(
            f"User asked:\n```{user_message}```\n"
            "Rewrite this into a brief search query (just a few keywords, no explanation)."
        )
    )

    # Make the call
    response = llm.invoke([query_system, query_user])

    # Check if it's a message object (expected) or a list (error case)
    if hasattr(response, "content"):
        if isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response.content, list):
            # Join list elements into a string
            return " ".join(str(item) for item in response.content).strip()
        else:
            return str(response.content).strip()
    else:
        raise ValueError("❌ LLM did not return a valid message object.")

@app.post("/chat")
async def chat(input: ChatInput):
    internal_docs_text = ""
    tavily_text = ""
    formatted_output_docs = ""

    if input.usedb:
        try:
            search_query = generate_search_query(input.message)
            db_collection = input.db or os.getenv("ASTRA_DB_COLLECTION", "files")
            retriever = get_vector_store(db_collection)\
                            .as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(search_query)
            unique_docs = deduplicate_docs(retrieved_docs)
            internal_docs_text = format_as_message(unique_docs, mode="openai")
            formatted_output_docs = format_as_message(unique_docs, mode="output")
        except Exception as e:
            return {"error": f"❌ DB retrieval failed: {e}"}

    # 2. Optionally retrieve Tavily web results
    if input.useweb:
        tavily_result = tavily_search(
            query=input.message,
            search_depth="basic",
            chunks_per_source=3,
            topic="general",
            max_results=3,
            include_answer=True,
            include_images=False,
            include_raw_content=False
        )
        tavily_text = (tavily_result.get("answer") or "").strip()

    # 3. Retrieve Google search results (only used in output, not prompt)
    google_results: List[str] = []
    if input.useweb:
        try:
            google_results = google_search(query=input.message, k=3)
        except Exception as e:
            google_results = [f"❗️Google Search Error: {str(e)}"]

    # 4. Construct Structured Prompt with Role-Based Blocks
    prompt_parts = []
    
    system_prompt = (
    "You are 桜AI, an AI assistant for パシフィックコンサルタンツ株式会社 (PCKK)."
    "You help answer internal company questions using retrieved internal docs or web results. "
    "If the user needs help beyond the AI's response or wants to consult on individual cases, direct them to the appropriate DX事業推進部 contact:\n\n"
    "【地方別担当】\n"
    "・北海道：今井\n・東北：谷口\n・北関東：吉川\n・南関東：中央省庁の分野担当に準拠\n"
    "・北陸：札本\n・中部：道端\n・関西：轟\n・中国：保多\n・四国：野添\n・九州：新井\n\n"
    "【中央省庁の分野担当】\n"
    "・国土交通省 - 観光庁：今井・保多\n"
    "・国土交通省 - 総合政策局 / 鉄道局：北野・新井\n"
    "・国土交通省 - 物流・自動車局：金木・保多・道端\n"
    "・国土交通省 - 都市局 / 道路局：札本\n"
    "・国土交通省 - 不動産・建設経済局 / 住宅局：轟\n"
    "・国土交通省 - 航空局 / 北海道局：今井\n"
    "・総務省：今川\n・経済産業省：吉川\n・環境省：今井")

    # [Context] section
    if internal_docs_text:
        prompt_parts.append("[Context: Internal Documents]\n" + internal_docs_text)
    if tavily_text:
        prompt_parts.append("[Context: Web Results]\n" + tavily_text)
        
    # [User Question]
    prompt_parts.append(f"[User Question]\n{input.message}")

    # Final instruction
    prompt_parts.append("[Answer]")

    final_prompt = "\n\n".join(prompt_parts)
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1.0
    )
    
    # 5. Send to LLM with structured role-based messages
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(convert_history_to_messages(input.history))  # Add history
    messages.append(HumanMessage(content=final_prompt))     # Current user input
    gpt_reply = llm.invoke(messages)


    # 6. Final output
    final_output = str(gpt_reply.content).strip()

    # Contact line
    contact_line = "\nご不明な点がございましたら、以下のアドレスまでお気軽にお問い合わせください。\n" \
                   "[future-service-devlopment@tk.pacific.co.jp](mailto:future-service-devlopment@tk.pacific.co.jp)\n"

    # Append internal docs and contact info
    if input.usedb and formatted_output_docs:
        final_output += "\n\n### 社内文書情報:\n\n" + formatted_output_docs
        final_output += contact_line

    # Append web info
    if input.useweb and (google_results or tavily_text):
        final_output += "\n\n### オンラインWeb情報:\n"

        if google_results:
            final_output += "\n" + "\n".join(google_results)

        if tavily_text:
            final_output += "\n" + tavily_text

    return {
        "reply": final_output
    }


"""
return {
    "reply": gpt_reply,
    "retrieved_docs": docs_content if input.usedb else [],
    "formatted_output_docs": formatted_output_docs if input.usedb else None,
    "used_tavily": input.useweb,
    "tavily_text": tavily_text if input.useweb else None,
    "google_results": google_results if input.useweb else None
}
"""

from extractor_agent import run_extraction_agent
from uuid import uuid4
from datetime import datetime
import json
import re
#from astra_retriever import vector_store

def get_vector_store(collection: str):
    return AstraDBVectorStore(
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        namespace=os.getenv("ASTRA_DB_NAMESPACE"),
        collection_name=collection,
        autodetect_collection=True
    )

class ExtractInput(BaseModel):
    input: str
    session_id: str
    db: str

def extract_url(text: str) -> str | None:
    """Extract the first URL from the text if present."""
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None

@app.post("/skmod")
async def extract(input: ExtractInput):
    # Step 1: Run the extraction agent
    structured_result = run_extraction_agent(input.input)

    # ✅ Ensure it's a string
    if not isinstance(structured_result, str):
        try:
            structured_result = json.dumps(structured_result, ensure_ascii=False, indent=2)
        except Exception:
            structured_result = str(structured_result)

    # Step 2: Generate a unique message ID
    msgid = str(uuid4())

    # Step 3: Check if input has a URL
    url_in_input = extract_url(input.input)

    # Step 4: Build metadata with URL or session_id
    metadata = {
        "msgid": msgid,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if url_in_input:
        metadata["url"] = url_in_input
    else:
        metadata["session_id"] = input.session_id

    # Step 5: Attempt to save to Astra DB

    db_collection = input.db or os.getenv("ASTRA_DB_COLLECTION", "files")

    try:
        vector_store = get_vector_store(db_collection)
        vector_store.add_texts(
            texts=[structured_result],
            metadatas=[metadata]
        )
    except Exception as e:
        print("❌ Error during Astra DB save:", e)
        return {"error": f"❌ Astra DB save failed: {str(e)}"}

    # Step 6: Return formatted confirmation response
    return {
        "reply": (
            "✅ メッセージを保存しました！ 🎉✨\n\n"
            "📂 抽出データ:\n\n"
            f"{structured_result}\n\n"
            "🗑️ このメッセージを削除すると、抽出データも一緒に消えちゃうよ！\n\n"
            f"MSGID: {msgid}\n\n"
        )
    }