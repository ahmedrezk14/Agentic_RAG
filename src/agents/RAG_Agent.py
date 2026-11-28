import asyncio
import json
from typing import TypedDict, Annotated, Literal, Any, Dict, Optional

from  langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from controllers import NLPController
from models.ProjectModel import ProjectModel

MAX_RETRIES=2
MAX_GEN_RETRIES=2

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    project_id: int
    query: str
    queries: list[str]
    filters: Optional[Dict[str,Any]]
    extracted_filters: Optional[Dict[str,Any]]
    route : Literal["retrieve", "direct", "clarify"]
    documents : list[str] 
    is_relevant: bool
    retry_count: int
    is_grounded: bool
    gen_retry_count: int
    answer: str

def _call_llm(generation_client, system_prompt: str, user_prompt:str) -> str:
    """Invoke the generation client with a fresh system-prompted chat history."""

    chat_history=[
        generation_client.construct_prompt(
            prompt=system_prompt,
            role=generation_client.enums.SYSTEM.value
        )
    ]
    response=generation_client.generate_text(
        prompt=user_prompt,
        chat_history=chat_history
    )

    return (response or "").strip()

def _parse_json(text: str, default: dict)-> dict:
    """Parse a JSON object from LLM output, stripping optional markdown fences."""
    
    cleaned=text.strip()
    if cleaned.startswith("```"):
        lines=cleaned.split('\n')
        cleaned='\n'.join(lines[1:-1] if lines[-1].strip()=='```' else lines[1:]).strip()
    try:
        data=json.loads(cleaned)
        return data if isinstance(data, dict) else default
    except json.JSONDecodeError:
        return default

def _format_history(messages:list[AnyMessage], max_turns:int=5, max_chars: int=4000)->str:
    """Format recent prior conversation turns within a character budget."""

    prior=messages[:-1][-max_turns:]
    if not prior:
        return "None"
    
    lines=[]
    total_chars= 0
    for msg in reversed(prior):
        role = "User" if isinstance(msg,HumanMessage) else "Assistant"
        entry=f"{role}:\n{str(msg.content)[:1500]}"
        if total_chars+len(entry)>max_chars:
            break
        lines.append(entry)
        total_chars+=len(entry)

    return "\n".join(reversed(lines)) or "None"


async def search_documents(
    nlp_controller: NLPController,
    project,
    query: str, 
    limit: int = 8,
    metadata_filter:Optional[Dict[str,Any]]=None
)-> list[str]:
    """Run hybrid vector search with reranking and return tagged chunk strings."""

    results=await nlp_controller.search_vector_db_collections(
        project= project,
        text= query,
        limit= limit,
        search_mode = "hybrid",
        use_reranker= True,
        rerank_top_k= 25,
        rerank_score_threshold= -1.0,
        metadata_filter=metadata_filter
    )
    
    if not results:
        return []

    return [f"[id={doc.chunk_id}]\n{doc.text}"for doc in results]


def build_rag_graph(checkpointer, nlp_controller:NLPController, db_client):
    """Build and compile the adaptive and corrective RAG LangGraph workflow."""

    generation_client=nlp_controller.generation_client

    async def contextualize_and_route (state:AgentState) -> dict:
        """Contextualize follow-up messages, decompose into sub-queries, and select a route."""

        last_query=str(state["messages"][-1].content)
        history_text=_format_history(state["messages"])
        
        system_prompt = (
            "You are a query analyzer for a document assistant. "
            "Given conversation history and the latest user message:\n"
            "1. Set 'query': rewrite the latest message into a standalone question ONLY if it contains "
            "pronouns (e.g., it, they, that, those) or implicit references to prior turns; "
            "otherwise keep the latest message verbatim.\n"
            "2. Set 'queries': produce 1 to 3 concise search sub-queries from 'query' "
            "(1 for simple questions; 2-3 for multi-part or comparison questions).\n"
            "3. Set 'route' using these exact rules:\n"
            '   - "direct": greetings, thanks, farewells, or small talk not needing documents.\n'
            '   - "clarify": ambiguous or underspecified document requests missing key context '
            "(such as which document, party, or topic is meant).\n"
            '   - "retrieve": specific questions about document content, clauses, facts, or comparisons.\n'
            "4. Set 'filters': use ONLY the \"source\" key when the user explicitly names a specific file, "
            'source, or contract/company identifier (e.g., {"source": "agreement.pdf"} or '
            '{"source": "VitalibisInc"}); never invent other metadata keys, and return {} if no '
            "specific source or contract name is explicitly stated.\n"
            'Respond ONLY with valid JSON (no markdown, no explanation): '
            '{"query": "...", "queries": ["..."], "route": "retrieve|direct|clarify", "filters": {}}'
        )        

        user_prompt=(
            f"Conversation History:\n{history_text}"  
            f"\n\nLast message: {last_query}"     
        )
        
        raw = await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )
        parsed=_parse_json(raw, {"query" : last_query, "queries":[last_query] , "route": "retrieve", "filters":{}})

        route=parsed.get("route", "retrieve")

        if route not in ("retrieve", "clarify", "direct"):
            route="retrieve"
        
        standalone_query=str(parsed.get("query") or last_query).strip()
        raw_queries=parsed.get("queries")
        sub_queries=[]
        if isinstance(raw_queries,list):
            sub_queries=[str(q).strip() for q in raw_queries if q is not None and str(q).strip()][:3]
        if not sub_queries:
            sub_queries=[standalone_query]

        raw_filters=parsed.get('filters')
        extracted_filters=(
            {
                str(k):v for k,v in raw_filters.items() if k and v not in (None,"")
            }
            if isinstance(raw_filters,dict)
            else {}
        ) or None
        
        return {
            'query': standalone_query,
            'queries': sub_queries,
            'extracted_filters':extracted_filters,
            'route': route,
            'documents': [],
            'is_relevant': False,
            'retry_count': 0,
            'is_grounded': False,
            'gen_retry_count': 0,
        }
    
    async def retrieve_docs(state: AgentState) -> dict:
        """Fetch deduplicated top chunks concurrently across all decomposed sub-queries."""

        project_model=await ProjectModel.create_instance(db_client=db_client)
        project=await project_model.get_project_or_create_one(project_id=state['project_id'])
        
        queries=state.get('queries') or [state['query']]

        extracted_filters=state.get("extracted_filters") or {}
        explicit_filters=state.get("filters") or {}

        merged_filters={}
        if state.get('retry_count',0)>0:
            merged_filters=dict(explicit_filters) or None
        else :
            merged_filters={**extracted_filters,**explicit_filters} or None

        async def _run_queries(current_filters):
            batches=await asyncio.gather(
                *[
                    search_documents(
                        nlp_controller= nlp_controller,
                        project=project,
                        query= q,
                        limit= 8,
                        metadata_filter=current_filters
                    )
                    for q in queries
                ]
            )

            return list(
                dict.fromkeys(
                    chunk 
                    for batch in batches
                    for chunk in batch
                )
            )[:8]

        unique_chunks= await _run_queries(merged_filters)

        if not unique_chunks and merged_filters is not None:
            unique_chunks=await _run_queries(None)
        
        chunks=[
            chunk.replace('[id=',f'[Chunk {idx} | id=',1)
            for idx,chunk in enumerate(unique_chunks,start=1)
        ]

        return{'documents': chunks}

    async def judge_relevant(state: AgentState) -> dict:
        """Grade whether retrieved chunks contain evidence to answer the query."""

        if not state['documents']: # no documents found
            return {'is_relevant': False}

        system_prompt = (
            "You are a relevance grader for retrieved document chunks. "
            "Mark 'is_relevant' as true if the chunks contain useful information that helps answer "
            "any part of the question (even partial evidence for multi-part questions); "
            "mark false only if none of the chunks contain relevant facts.\n"
            'Respond ONLY with valid JSON (no markdown): {"is_relevant": true} or {"is_relevant": false}'
        )

        chunks_block="\n\n".join(state["documents"])
        user_prompt= (
            f"Question: {state['query']}"
            f"\n\nRetrieved Chunks:\n{chunks_block}"
        )   

        raw = await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )

        parsed=_parse_json(raw, {"is_relevant": False})

        return {"is_relevant" : bool(parsed.get("is_relevant", False))}
        
    async def rewrite_query(state: AgentState) -> dict:
        """Rephrase failed search queries with domain synonyms and expanded acronyms."""

        system_prompt = (
            "You optimize failed search queries for hybrid keyword and semantic retrieval. "
            "Generate 1 to 3 alternative search queries by expanding abbreviations or acronyms "
            "and adding relevant legal, contractual, or domain synonyms.\n"
            'Respond ONLY with valid JSON (no markdown): {"queries": ["..."]}'
        )

        user_prompt=(
            f"Original question: {state['query']}\n"
            f"Previous queries: {state.get('queries',[state['query']])}"
        )

        raw= await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )

        parsed=_parse_json(raw,{'queries':[state['query']]})

        raw_queries=parsed.get('queries')

        new_queries=(
            [ str(q).strip() for q in raw_queries if q is not None and str(q).strip() ][:3]
            if isinstance(raw_queries,list)
            else []
        )

        return {
            "queries": new_queries or [state["query"]],
            "retry_count": state["retry_count"] + 1,
            "extracted_filters":None
        }

    async def generate_rag_answer(state: AgentState)->dict:
        """Generate a cited answer strictly from the retrieved context chunks."""

        history_text=_format_history(state['messages'])
        chunks_block="\n\n".join(state['documents'])

        strict_note = ""
        if state.get("gen_retry_count", 0) > 0:
            strict_note = (
                " STRICT RETRY: Your previous draft included unsupported claims. "
                "Include ONLY facts, figures, and terms explicitly stated in the chunks."
            )

        system_prompt = (
            "You are a precise document assistant. Answer the question using ONLY the provided context chunks.\n"
            "Rules:\n"
            "1. Cite supporting chunks inline immediately after each claim using standard ASCII square brackets ONLY "
            "(for example: [Chunk 1], [Chunk 2]). Never use full-width brackets like 【Chunk 1】.\n"
            "2. Entity verification: Before using or citing any chunk, verify it belongs to the exact contracting "
            "parties, company, or agreement named in the Question. Ignore chunks from unrelated contracts; if no "
            "retrieved chunk matches the named party or agreement, state clearly that the requested agreement was not found.\n"
            "3. Do not infer, extrapolate, or add outside knowledge.\n"
            "4. If the chunks only partially answer the question, state what is supported and note what is missing.\n"
            "5. CRITICAL: Detect the language of the Question and respond strictly in that exact same language "
            "(e.g., English for English, Arabic for Arabic). Never switch to Chinese or any other language." + strict_note
        )

        user_prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Context chunks:\n{chunks_block}\n\n"
            f"Question: {state['query']}\n"
            "Answer (strictly in the exact same language as the Question):"
        )

        answer=await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )

        if not answer:
            answer="I couldn't generate an answer from the retrieved documents."

        return {"answer": answer}
    
    
    async def check_groundedness(state: AgentState)-> dict:
        """Verify that every factual claim in the draft answer is supported by chunks."""

        system_prompt = (
            "You are a factual grounding verifier. Verify that every factual claim in the answer is supported by, "
            "accurately paraphrased from, or directly synthesized from facts explicitly stated across the context chunks.\n"
            "Do NOT reject valid paraphrasing, combining explicit facts from multiple chunks, transition phrases, "
            "inline citation tags ([Chunk X] or 【Chunk X】), or statements noting that a detail or contract was not found.\n"
            "Return false ONLY if the answer contradicts the chunks or introduces fabricated/unsupported facts, "
            "numbers, dates, party names, or legal conclusions.\n"
            'Respond ONLY with valid JSON (no markdown): {"is_grounded": true} or {"is_grounded": false}'
        )

        chunks_block="\n\n".join(state['documents'])
        user_prompt=(
            f"Context chunks:\n{chunks_block}\n\n"
            f"Answer:\n{state['answer']}"
        )

        raw=await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )
        
        parsed=_parse_json(raw,{'is_grounded':True})
        is_grounded=bool(parsed.get('is_grounded',True))
        
        if is_grounded:
            return {
                'is_grounded':True,
                'messages':[AIMessage(content=state['answer'])]
            }
        return {
            'is_grounded':False,
            'gen_retry_count':state.get('gen_retry_count',0)+1
        }

    def decision_after_groundedness(state: AgentState)->Literal["end","regen","fallback"]:
        """Route to completion, regeneration, or fallback based on groundedness."""

        if state['is_grounded']:
            return 'end'
        if state['gen_retry_count']<=MAX_GEN_RETRIES:
            return 'regen'
        return 'fallback'

    async def answer_without_docs(state: AgentState)->dict:
        """Respond directly to conversational queries without document retrieval."""

        history_text=_format_history(state['messages'])
        
        system_prompt = (
            "You are a polite, concise document assistant handling general conversation or greetings. "
            "CRITICAL: Detect the language of the User message and reply naturally in that exact same language "
            "(e.g., English for English, Arabic for Arabic). Never use any other language, never invent or guess "
            "facts about their documents, and invite them to ask a document-specific question when helpful."
        )
        
        user_prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"User message: {state['query']}\n"
            "Reply (in the exact same language as the User message):"
        )
    
        answer=await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )

        if not answer:
            answer="Hello! How can I help you with your documents today?"

        return {
            "answer": answer,
            "messages":[AIMessage(content=answer)]
        }

    async def ask_clarify(state: AgentState) -> dict:    
        """Ask a concise clarifying question for ambiguous user requests."""

        system_prompt = (
            "The user's document request is ambiguous or missing key context. "
            "Ask ONE brief, polite clarifying question (for example, which document, party, or clause "
            "they are asking about) without guessing an answer. "
            "CRITICAL: Detect the language of the User message and ask your question strictly in that exact same language "
            "(e.g., English for English, Arabic for Arabic). Never use any other language."
        )

        user_prompt = (
            f"User message: {state['query']}\n"
            "Clarifying question (in the exact same language as the User message):"
        )

        answer=await asyncio.to_thread(
            _call_llm,
            generation_client,
            system_prompt,
            user_prompt
        )
    
        if not answer:
            answer="Could you clarify what specific topic or document detail you are asking about?"
        
        return {
            "answer": answer,
            "messages":[AIMessage(content=answer)]
        }

    async def fallback_not_found(state: AgentState)-> dict:
        """Return a safe fallback response when no grounded answer is found."""

        answer="I couldn't find any relevant information in the documents."
        return {
            "answer" : answer ,
            "messages" : [AIMessage(content=answer)],
        }
    
    def route_query(state: AgentState) -> Literal['retrieve', 'clarify', 'direct']:
        """Return the selected adaptive route key for conditional graph branching."""

        return state['route']
    
    def decision_after_judgement(state: AgentState)-> Literal['generate', 'fallback', 'rewrite']:
        """Route to answer generation, query rewriting, or fallback after grading."""

        if state['is_relevant']:
            return 'generate'
        if state['retry_count'] < MAX_RETRIES:
            return 'rewrite'
        return 'fallback'

    
    graph=StateGraph(AgentState)

    #### Graph nodes ####
    graph.add_node('contextualize_and_route',contextualize_and_route)
    graph.add_node('retrieve_docs',retrieve_docs)
    graph.add_node('judge_relevant',judge_relevant)
    graph.add_node('rewrite_query',rewrite_query)
    graph.add_node('generate_rag_answer',generate_rag_answer)
    graph.add_node('answer_without_docs',answer_without_docs)
    graph.add_node('ask_clarify',ask_clarify)
    graph.add_node('fallback_not_found',fallback_not_found)
    graph.add_node('check_groundedness',check_groundedness)

    #### Graph edges ####
    
    graph.add_edge(START,'contextualize_and_route')
    graph.add_conditional_edges(
        'contextualize_and_route',
        route_query,
        {
            'retrieve':'retrieve_docs', 
            'clarify':'ask_clarify', 
            'direct':'answer_without_docs',
        },
    )
    graph.add_edge('retrieve_docs','judge_relevant')
    graph.add_conditional_edges(
        'judge_relevant',
        decision_after_judgement,
        {
            'generate':'generate_rag_answer',
            'fallback':'fallback_not_found',
            'rewrite':'rewrite_query',
        },
    )
    graph.add_edge('rewrite_query','retrieve_docs')
    graph.add_edge('generate_rag_answer','check_groundedness')
    graph.add_conditional_edges(
        'check_groundedness',
        decision_after_groundedness,
        {
            'end':END,
            'regen':'generate_rag_answer',
            'fallback':'fallback_not_found'
        }
    )
    
    graph.add_edge('fallback_not_found',END)
    graph.add_edge('ask_clarify',END)
    graph.add_edge('answer_without_docs',END)

    return graph.compile(checkpointer=checkpointer)

    