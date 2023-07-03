from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType


EXLLAMA_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
    "### RESPONSE:\n"
    
)
EXLLAMA_REFINE_PROMPT = Prompt(
    EXLLAMA_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)

EXLLAMA_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n### RESPONSE:\n"
)
EXLLAMA_TEXT_QA_PROMPT = Prompt(
    EXLLAMA_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)