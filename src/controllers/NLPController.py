from .BaseController import BaseController
from models.db_schemes import Project,DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Literal, Optional, Dict, Any
import json
from .RetrievalFusion import reciprocal_rank_fusion

class NLPController(BaseController):
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser, reranker_client=None):
        super().__init__()
        self.vectordb_client=vectordb_client
        self.generation_client=generation_client
        self.embedding_client=embedding_client
        self.template_parser=template_parser
        self.reranker_client = reranker_client

    def create_collection_name(self, project_id:str):
        return f"collection_{project_id}_{self.vectordb_client.default_vector_size}".strip()

    async def reset_vector_db_collection(self, project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        collection_info=await self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(
            json.dumps(collection_info,default=lambda x:x.__dict__)
        )

    async def index_into_vector_db(self, project:Project, chunks:List[DataChunk], chunk_ids:list[int], do_reset:bool=False):
        # 1 -> get collection name 
        collection_name=self.create_collection_name(project_id=project.project_id)

        # 2 -> hendel data 
        texts=[chunk.chunk_text for chunk in chunks]
        metadata=[chunk.chunk_metadata for chunk in chunks]
        vectors=self.embedding_client.embed_text(
                        text=texts,
                        document_type=DocumentTypeEnum.DOCUMENT.value
                    )

        # 3 -> create collection 
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )
        
        # 4 -> insert in df 
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            record_ids=chunk_ids
        )

        return True

    async def search_vector_db_collections(
        self,
        project: Project,
        text: str,
        limit: int = 10,
        search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: bool = False,
        rerank_top_k: int = 25,
        rerank_score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str,Any]]= None
    ):
        
        # 1-> get collection name 
        collection_name=self.create_collection_name(project_id=project.project_id)

        # 2-> determine candidate limits for reranking and hybrid fusion
        rerank=use_reranker and self.reranker_client is not None
        retrieval_limit=max(rerank_top_k,limit) if rerank else limit

        if search_mode=='sparse':
            results=await self.vectordb_client.sparse_search(
                collection_name=collection_name,
                text=text,
                limit=retrieval_limit,
                metadata_filter=metadata_filter
            )
            if not results :
                return False
            if rerank:
                return self.reranker_client.rerank(
                    query=text,
                    documents=results,
                    top_k=limit,
                    score_threshold=rerank_score_threshold,
                )
            return results
        
        candidate_limit=max(retrieval_limit*5,25) if search_mode =='hybrid' else retrieval_limit
        query_vector=None
        vectors=self.embedding_client.embed_text(text=text,document_type=DocumentTypeEnum.DOCUMENT.value)
        if not vectors or len(vectors)==0:
            return False
        if isinstance(vectors, list):        
            query_vector=vectors[0]
        if query_vector is None:
            return False

        # 3 -> semantic search
        dense_results=await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=candidate_limit,
            metadata_filter=metadata_filter
        )
        if search_mode=='dense':
            if not dense_results:
                return False
            if rerank:
                return self.reranker_client.rerank(
                    query=text,
                    documents=dense_results,
                    top_k=limit,
                    score_threshold=rerank_score_threshold,
                )
            return dense_results
        sparse_results = await self.vectordb_client.sparse_search(
            collection_name=collection_name,
            text=text,
            limit=candidate_limit,
            metadata_filter=metadata_filter
        )
        # 4-> hybrid search: combine dense and sparse candidates via RRF
        fused_results=reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=60,
            limit=retrieval_limit,
            w_dense=1.3,
            w_sparse=0.7,
        )

        if not fused_results:
            return False
        if rerank:
            return self.reranker_client.rerank(
                query=text,
                documents=fused_results,
                top_k=limit,
                score_threshold=rerank_score_threshold,
            )
        return fused_results

    async def answer_rag_question(
        self,
        project:Project,
        query:str, 
        limit:int=10,
        search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        use_reranker: bool = False,
        rerank_top_k: int = 25,
        rerank_score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str,Any]]= None
    ):
        
        answer, full_prompt, chat_history=None,None,None

        # 1-> retrive relevant docs  
        retrived_documents=await self.search_vector_db_collections(
            project=project,
            text=query,
            limit=limit,
            search_mode=search_mode,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
            rerank_score_threshold=rerank_score_threshold,
            metadata_filter=metadata_filter
        )

        if not retrived_documents or len (retrived_documents) == 0:
            return answer, full_prompt, chat_history

        # 2-> build prompt

        sysyem_prompt=self.template_parser.get(group="rag", key="system_prompt", vars={})

        document_prompts="\n".join([
            self.template_parser.get(group="rag", key="document_prompt", vars={
                "doc_num":idx+1,
                "chunk_text":self.generation_client.process_text(document.text)
            })
            for idx,document in enumerate(retrived_documents)
        ])

        footer_prompt=self.template_parser.get(group="rag", key="footer_prompt", vars={
            "query": query
        })
        
        chat_history=[
            self.generation_client.construct_prompt(
                prompt=sysyem_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt="\n\n".join([document_prompts,footer_prompt])

        answer = self.generation_client.generate_text(prompt=full_prompt,chat_history=chat_history)

        return answer, full_prompt, chat_history



