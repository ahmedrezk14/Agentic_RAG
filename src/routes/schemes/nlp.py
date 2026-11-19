from pydantic import BaseModel
from typing import Literal, Optional

class PushRequest(BaseModel):
    do_reset:Optional[int]=0

class SearchhRequest(BaseModel):
    text:str
    limit:Optional[int]=10
    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    use_reranker: bool = True
    rerank_top_n: int = 20
