from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from newsapi import NewsApiClient
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field



mem = MemorySaver()
load_dotenv()

class AgentState(TypedDict):
    query: str
    keywords: list[str]
    currentKeyword: int
    numKeywords: int
    articles: list[str]
    summaries: list[str]
    final: Optional[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

class KeywordOutput(BaseModel):
    keywords: List[str] = Field(description="List of extracted concise search keywords")

keyword_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent search optimizer.  
Given this user input:

"{query}"

Extract 1–4 important search keywords or short phrases that capture its core meaning.  
Include related or synonymous terms if helpful.

Return them as a list of concise keywords like:
["keyword1", "keyword2", "keyword"]
"""
)

llmStructured = llm.with_structured_output(KeywordOutput)

keyword_chain = keyword_prompt | llmStructured

def keyword_extraction_node(state: AgentState):
    """
    Analyzes user input and extracts the most relevant search keywords.
    Returns both the keywords list and a refined query string.
    """
    try:
        query = state.get("query", "")
        print(f"🔍 Extracting keywords from query: {query}")
        response = keyword_chain.invoke({"query": query})
        keywords = response.keywords
        print(f"🔍 Extracted keywords: {keywords}")
        state["keywords"] = keywords
        state["numKeywords"] = len(keywords)
        state["currentKeyword"] = 0  # Initialize counter
        state["articles"] = []       # Prepare container
        state["summaries"] = []      # Prepare container
        print(f"Extracted {len(keywords)} keywords:", keywords)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
    
    return state


newsapi = NewsApiClient(api_key="c7e47f3fdfb64c5eb86661d323d92fbc")

def get_news(query: str):
    """Fetches recent English news articles related to a query keyword."""
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy")
    return [
        {"title": a["title"], "url": a["url"], "description": a["description"]}
        for a in articles["articles"][:5]  # only first 5 results
    ]

# Define a simple prompt for summarization
prompt = PromptTemplate(
    input_variables=["title", "description"],
    template=(
        "Summarize this news article clearly and concisely in 5 sentences:\n\n"
        "Title: {title}\n\n"
        "Description: {description}\n\n"
        "Summary:"
    )
)
summarizer_chain = prompt | llm

def summarize_article(article):
    """Summarizes a single article using the LLM."""
    try:
        summary = summarizer_chain.invoke({
            "title": article.get("title", ""),
            "description": article.get("description", "")
        })
        return {
            "title": article["title"],
            "url": article["url"],
            "summary": summary.content.strip(),
        }
    except Exception as e:
        return {
            "title": article["title"],
            "url": article["url"],
            "summary": f"Error summarizing article: {e}",
        }
    
def fetch_and_summarize(state: AgentState):
    """Fetches 5 recent articles and summarizes them in parallel."""
    # Use .invoke to get structured Python data
    keywords = state.get("keywords") or state.get("Keywords")
    if not keywords:
        print("⚠️ No keywords found in state. Skipping fetch.")
        return state

    current_index = state.get("currentKeyword", 0)
    query = keywords[current_index]

    articles = get_news(query)
    state["articles"].append(articles)
    print(f"Fetched {len(articles)} articles.")
    state["currentKeyword"] += 1
    print(f"✅ Processed keyword {current_index+1}/{len(keywords)}: {query}")

    # Run summaries in parallel for speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(summarize_article, a): a for a in articles}
        for future in as_completed(futures):
            state["summaries"].append(future.result())

    return state


merge_prompt = PromptTemplate(
    input_variables=["summaries"],
    template="""
You are a professional news editor tasked with combining several summaries of different articles 
about the same topic into a single, seamless and natural summary.

Your output should:
- Merge overlapping ideas smoothly (avoid listing each article separately)
- Preserve unique insights from each summary
- Maintain a consistent neutral tone
- Read like one continuous piece of writing, not a stitched list

Here are the summaries to merge:
{summaries}

Now write the final unified summary  dont make it short:
"""
)

merger_chain =  merge_prompt | llm


def mergeSummariesLLM(state: AgentState):
    """
    Takes a list of individual article summaries and merges them seamlessly with an LLM.
    """
    formatted = "\n\n".join(
        [f"Summary {i+1}:\n{summary}" for i, summary in enumerate(state["summaries"])]
    )
    merged = merger_chain.invoke({"summaries": formatted})
    state["final"] = merged.content
    return state

bias_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
You are an expert media analyst. Analyze the following unified news summary for tone and bias.

Text:
{summary}

Provide a brief structured report including:
- Overall sentiment (Positive / Negative / Neutral)
- Political or ideological bias (if any)
- Presence of emotionally charged language
- Balance of perspectives
- Confidence score (0–100%)

Respond in JSON format:
{{
  "sentiment": "",
  "bias": "",
  "emotion_level": "",
  "balance": "",
  "confidence": ""
}}
"""
)


class BiasAnalysisOutput(BaseModel):
    """Structured bias/sentiment analysis output."""
    sentiment: str = Field(..., description="Overall sentiment: Positive, Negative, or Neutral.")
    bias: str = Field(..., description="Political or ideological bias if any.")
    emotion_level: str = Field(..., description="Degree of emotionally charged language.")
    balance: str = Field(..., description="Assessment of balance across perspectives.")
    confidence: str = Field(..., description="Confidence score as a percentage.")

llmStructured = llm.with_structured_output(BiasAnalysisOutput)

bias_chain = bias_prompt | llmStructured


def output_node(state: AgentState):
    bias_report = bias_chain.invoke({"summary": state["final"]})
    res = bias_report["text"] if isinstance(bias_report, dict) else bias_report
    state["final"] = state["final"] + "\n\nBias Analysis Report:\n" + str(res)
    return state


def router(state: AgentState):
    current = state.get("currentKeyword", 0)
    total = state.get("numKeywords", 0)
    if current < total:
        return "fetch_summarize"
    return "merge_summaries"

graph = StateGraph(AgentState)

graph.add_node("keyword", keyword_extraction_node)
graph.add_node("fetch_summarize", fetch_and_summarize)
graph.add_node("merge_summaries", mergeSummariesLLM)
graph.add_node("output", output_node)

graph.add_edge(START, "keyword")
graph.add_edge("keyword", "fetch_summarize")
graph.add_conditional_edges("fetch_summarize",
                             router,
                               {"fetch_summarize": "fetch_summarize",
                                "merge_summaries": "merge_summaries"
                               })
graph.add_edge("merge_summaries", "output")
graph.add_edge("output", END)

app = graph.compile()

result = app.invoke({
            "query": [HumanMessage(content="AI in motorsports")]
        })
print(result["final"])