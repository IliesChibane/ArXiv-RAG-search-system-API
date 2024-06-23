from fastapi import FastAPI
from pydantic import BaseModel
from rag_utils import run_arxiv_search

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    num_results: int

@app.post("/process-query")
async def get_results(request: QueryRequest):
    query_text = request.query
    num_results_to_print = request.num_results

    return run_arxiv_search(query_text, num_results_to_print)