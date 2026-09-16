from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a research assistant that answers questions strictly \
based on the provided evidence chunks. Rules:

1. Only use information present in the evidence. Never invent facts.
2. Every claim you make MUST cite at least one evidence_id from the provided list.
3. If the evidence does not answer the question, say so explicitly in a claim \
   with an empty evidence_ids list and confidence "low".
4. Never invent an evidence_id that was not provided.
"""

REFINE_SYSTEM_PROMPT = """You rewrite research search queries. The previous \
query did not return enough useful evidence. Rewrite it to be more specific, \
use alternative terminology, or broaden/narrow scope as appropriate. \
Respond with ONLY the new query text, nothing else."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nEvidence:\n{evidence_block}"),
    ]
)

REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REFINE_SYSTEM_PROMPT),
        ("human", "Original question: {question}\nPrevious query: {previous_query}"),
    ]
)