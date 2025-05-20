from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

from astra_retriever import retriever
from message_formatter import format_as_message
from tavily_search import tavily_search
from google_search import google_search 

load_dotenv()

app = FastAPI()
llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

class ChatInput(BaseModel):
    message: str
    useweb: Optional[bool] = False  # 🔘 Toggle for Tavily/Google
    usedb: Optional[bool] = False  # 🔘 Toggle for Astra retriever

def deduplicate_docs(docs):
    unique = []
    seen = set()
    for doc in docs:
        identifier = (doc.page_content, frozenset(doc.metadata.items()) if hasattr(doc, 'metadata') else None)
        if identifier not in seen:
            unique.append(doc)
            seen.add(identifier)
    return unique

@app.post("/chat")
async def chat(input: ChatInput):
    formatted_context = ""
    docs_content = []
    formatted_output_docs = ""

    # 1. Retrieve internal documents via Astra (if enabled)
    if input.usedb:
        retrieved_docs = retriever.invoke(input.message)
        unique_docs = deduplicate_docs(retrieved_docs)
        formatted_context += format_as_message(unique_docs, mode="openai")
        formatted_output_docs = format_as_message(unique_docs, mode="output")
        docs_content = [doc.page_content for doc in unique_docs]


    # 2. Optionally retrieve Tavily web results
    tavily_text = ""
    if input.useweb:
        tavily_result = tavily_search(
            query=input.message,
            search_depth="advanced",
            chunks_per_source=3,
            topic="general",
            max_results=5,
            include_answer=True,
            include_images=False,
            include_raw_content=False
        )
        tavily_text = tavily_result.get("answer", "")
        if tavily_text:
            formatted_context += f"\n\nWeb Results:\n{tavily_text}"

    # 3. Retrieve Google search results (only used in output, not prompt)
    google_results = []
    if input.useweb:
        try:
            google_results = google_search(query=input.message, k=3)
        except Exception as e:
            google_results = [{"error": str(e)}]

    # 4. Compose final prompt with internal + tavily context (not Google)
    prompt = f"Context:\n{formatted_context}\n\nQuestion:\n{input.message}\nAnswer:"

    # 5. Run the LLM
    gpt_reply = llm.invoke([
        SystemMessage(content=""),
        HumanMessage(content=prompt)
    ])

    # 6. Format the final response
    final_output = gpt_reply.content.strip()

    if input.usedb and formatted_output_docs:
        final_output += "\n\n### 社内文書情報:\n\n" + formatted_output_docs

    if input.useweb and (google_results or tavily_text):
        final_output += "\n\n---\nご不明な点がございましたら、以下のアドレスまでお気軽にお問い合わせください。\n" \
                        "[future-service-devlopment@tk.pacific.co.jp](mailto:future-service-devlopment@tk.pacific.co.jp)\n"

        final_output += "\n\n### オンラインWeb情報:\n"

        if google_results:
            final_output += "\n" + "\n".join(google_results)


        if tavily_text:
            final_output += "\n---\n" + tavily_text.strip()

    return {
        "reply": final_output
    }


"""return {
        "reply": gpt_reply,
        "retrieved_docs": docs_content if input.usedb else [],
        "formatted_output_docs": formatted_output_docs if input.usedb else None,
        "used_tavily": input.useweb,
        "tavily_text": tavily_text if input.useweb else None,
        "google_results": google_results if input.useweb else None  # ✅ Output only
    }"""
