from pydantic import BaseModel
from typing import Optional,Literal, Any, Dict

class PushRequest(BaseModel):
    do_reset:Optional[int]=0

class SearchhRequest(BaseModel):
    text:str
    limit:Optional[int]=10
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    use_reranker: bool = True
    rerank_top_k: int = 25
    rerank_score_threshold: Optional[float] = None
    filters: Optional[Dict[str,Any]] = None

class AgentRequest(BaseModel):
    text: str
    thread_id: str
    filters: Optional[Dict[str,Any]] = None
