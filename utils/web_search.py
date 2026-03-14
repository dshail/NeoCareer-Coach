import requests
from config.config import TAVILY_API_KEY

def search_web(query, num_results=5):
    """
    Perform a web search using Tavily API.
    """
    if not TAVILY_API_KEY:
        return "Tavily API Key not found. Web search disabled."
    
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "smart",
            "include_answer": True,
            "max_results": num_results
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        results = response.json()
        
        snippets = []
        for result in results.get("results", []):
            snippets.append(f"Source: {result['url']}\nContent: {result['content']}")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error during web search: {str(e)}"

def llm_web_tool(query):
    """
    Wrapper for LLM to use as a tool (can be extended with summarization).
    """
    return search_web(query)
