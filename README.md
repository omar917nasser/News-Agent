# News Analysis Agent

This project implements a multi-agent autonomous system using LangGraph. The agent takes a user query, dynamically extracts relevant keywords, fetches news articles for each keyword, summarizes them in parallel, merges all summaries into a single cohesive report, and finally performs a bias and sentiment analysis on the combined text.

-----

## 🚀 Features

  * **Dynamic Keyword Extraction:** Uses a Google Gemini model to intelligently extract 1-4 relevant search keywords from a single user query.
  * **Parallel Fetch & Summarize:** For each keyword, it fetches the top 5 recent articles from the NewsAPI and summarizes them concurrently using a `ThreadPoolExecutor` for high speed.
  * **Cohesive Merging:** A dedicated "editor" agent merges the individual summaries into a single, seamless, and natural-sounding report, avoiding repetition.
  * **Bias & Sentiment Analysis:** A final "analyst" agent reads the merged summary and provides a structured JSON report on its sentiment, political bias, emotional language, and balance.
  * **Stateful Workflow:** Built on LangGraph, the entire process is a stateful graph, allowing for complex routing, loops, and robust state management.
  * **Structured Output:** Leverages Pydantic and LangChain's structured output capabilities to ensure reliable JSON and list outputs from the LLM.

-----

## ⚙️ How It Works

The project is built as a `StateGraph` with the following flow:

1.  **`START` -\> `keyword`**: The graph starts at the `keyword` node.

      * **Node:** `keyword_extraction_node`
      * **Action:** Takes the initial user query (e.g., "AI in motorsports") and uses an LLM to extract a list of search keywords (e.g., `["AI in racing", "motorsports machine learning", "Formula 1 AI"]`).
      * **State Update:** Populates `state['keywords']`, `state['numKeywords']`, and initializes counters.

2.  **`keyword` -\> `fetch_summarize`**: Moves to the first round of fetching.

      * **Node:** `fetch_and_summarize`
      * **Action:**
        1.  Takes the *first* keyword from the list.
        2.  Calls the NewsAPI to get the top 5 articles for that keyword.
        3.  Uses a `ThreadPoolExecutor` to summarize all 5 articles in parallel with an LLM.
        4.  Appends the new articles and summaries to the state.
      * **State Update:** Increments `state['currentKeyword']`.

3.  **`fetch_summarize` -\> `router` (Conditional Edge)**: After each run, the router checks the state.

      * **Logic:** `router` function
      * **Condition 1:** If `currentKeyword < numKeywords`, the graph is routed *back* to the `fetch_summarize` node to process the *next* keyword in the list.
      * **Condition 2:** If `currentKeyword == numKeywords` (all keywords processed), the graph is routed to the `merge_summaries` node.

4.  **`merge_summaries` -\> `output`**:

      * **Node:** `mergeSummariesLLM`
      * **Action:** Takes the (now complete) list of all summaries from `state['summaries']` and uses an "editor" LLM prompt to merge them into one single, high-quality summary.
      * **State Update:** Stores the result in `state['final']`.

5.  **`output` -\> `END`**:

      * **Node:** `output_node`
      * **Action:**
        1.  Takes the merged summary from `state['final']`.
        2.  Uses a "bias analyst" LLM prompt with structured Pydantic output to generate a JSON report on sentiment, bias, etc.
        3.  Appends this report to the final summary.
      * **State Update:** The `state['final']` now contains the complete report + analysis.
      * **Action:** The graph finishes and returns the final state.

-----

## 🛠️ Technology Stack

  * **Graph Framework:** [LangGraph](https://github.com/langchain-ai/langgraph)
  * **LLM:** Google Gemini (via `langchain-google-genai`)
  * **Core Libraries:** LangChain (LCEL, Prompts, Structured Output), Pydantic
  * **Data Source:** [NewsAPI](https://newsapi.org/)
  * **Observability:** [LangSmith](https://www.langchain.com/langsmith) (configured in `.env`)

-----

## 📦 Setup & Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2\. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3\. Install Dependencies

Create a `requirements.txt` file with the following contents:

```
langgraph
langchain
langchain-core
langchain-google-genai
newsapi-python
python-dotenv
pydantic
```

Then, install them:

```bash
pip install -r requirements.txt
```

### 4\. Set Up Environment Variables

Create a `.env` file in the root of the project and add your API keys.

```env
# Get from Google AI Studio
GOOGLE_API_KEY="AIzaSy..."

# Get from https://newsapi.org/
NEWSAPI_KEY="your_newsapi_key"

# Get from https://smith.langchain.com/ (Optional, for tracing)
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="lsv2_pt_..."
LANGSMITH_PROJECT="testParallel"
```

**Note:** The `app.py` file provided currently has a hardcoded NewsAPI key. It's recommended to modify the `newsapi` object initialization to use the environment variable for better security and flexibility:

```python
# Recommended change in app.py
import os
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
```

-----

## ▶️ How to Run

The `app.py` script is set up to run directly from the command line. The query is hardcoded at the bottom of the file.

### 1\. Modify the Query (Optional)

Open `app.py` and change the `content` in the `app.invoke` call at the very bottom:

```python
# Near the bottom of app.py
result = app.invoke({
            # Change this query to whatever you want
            "query": [HumanMessage(content="AI in motorsports")]
        })
print(result["final"])
```

### 2\. Run the Script

Execute the script from your terminal:

```bash
python app.py
```

The script will stream logs to the console, showing the keywords extracted, the articles fetched, and finally, it will print the complete, merged summary along with its bias analysis report.
