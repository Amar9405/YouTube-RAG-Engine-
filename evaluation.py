import sys
import types

# --- RAGAS compatibility shim (MUST BE AT THE VERY TOP) ---
_MODULE = "langchain_community.chat_models.vertexai"
if _MODULE not in sys.modules:
    _stub = types.ModuleType(_MODULE)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_MODULE] = _stub
# ----------------------------------------------------------

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_core.outputs import LLMResult

from ingestion import get_transcript, split_transcript
from retrieval import build_vector_store, get_multiquery_reranking_retriever, get_embeddings
from generation import build_rag_chain, get_llm

load_dotenv()

# --- SafeGroq Wrapper ---
class SafeGroq(ChatGroq):
    """Strips the 'n' parameter to prevent Groq 400 BadRequest errors."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

# 1. Our test questions and correct answers
questions = [
    "Who shot Osama bin Laden?",
    "Why did they not want to capture bin Laden alive?",
    "What happened to the helicopter?",
    "Why didn't they just bomb the house?",
    "What was the name of the dog?"
]
ground_truths = [
    "Rob O'Neill shot Osama bin Laden twice in the head.",
    "They wanted him dead because a trial would take too long.",
    "One helicopter crashed in the front yard during the hover.",
    "They needed to prove he was there and get DNA, so they couldn't just bomb it.",
    "The dog's name was Cairo."
]

def run_eval():
    print("⏳ Step 1: Loading video and building RAG chain...")
    vid = "sRH-XlBE2cQ"
    t = get_transcript(vid)
    c = split_transcript(t, vid)
    vs = build_vector_store(c)
    
    # Use your advanced retriever
    retriever = get_multiquery_reranking_retriever(vs)
    chain = build_rag_chain(retriever)
    
    print("\n🤖 Step 2: Generating answers for test questions...")
    answers = []
    contexts = []
    
    for q in questions:
        print(f"  -> Asking: {q}")
        try:
            ans = chain.invoke(q)
            answers.append(ans)
            docs = retriever.invoke(q)
            contexts.append([d.page_content for d in docs])
        except Exception as e:
            print(f"  Error on question: {e}")
            answers.append("")
            contexts.append([""])
        
    print("\n📊 Step 3: Running RAGAS Evaluation (Running 1 by 1 to avoid Groq Rate Limits. Please wait...)")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        # Use SafeGroq here!
        llm=LangchainLLMWrapper(SafeGroq(model="openai/gpt-oss-120b", temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        # RunConfig forces it to go one by one, preventing Rate Limits (Timeouts)
        run_config=RunConfig(max_workers=1, timeout=120)
    )
    
    print("\n✅ FINAL SCORES:")
    print(result.to_pandas())

if __name__ == "__main__":
    run_eval()