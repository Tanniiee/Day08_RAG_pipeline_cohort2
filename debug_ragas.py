"""Debug script — chạy: python debug_ragas.py"""
import sys, types, os
print(f"Python: {sys.version}")

# Patch vertexai stub
stub = types.ModuleType('langchain_community.chat_models.vertexai')
stub.ChatVertexAI = None
sys.modules['langchain_community.chat_models.vertexai'] = stub
try:
    import langchain_community.chat_models as parent; parent.vertexai = stub
except: pass

import warnings; warnings.filterwarnings("ignore")

from ragas.metrics.base import Metric
print(f"\nMetric base id: {id(Metric)}")

# 1. faithfulness submodule
print("\n=== faithfulness submodule ===")
import ragas.metrics.collections.faithfulness as fm
print(f"fm.metric type: {type(fm.metric)}")
print(f"isinstance(fm.metric, Metric): {isinstance(fm.metric, Metric)}")
print(f"fm.metric.llm: {getattr(fm.metric, 'llm', 'N/A')}")

# 2. What's in ragas.metrics.collections namespace
print("\n=== ragas.metrics.collections namespace ===")
import ragas.metrics.collections as rmc
items = [(k, type(v).__name__) for k, v in vars(rmc).items() if not k.startswith('_')]
for k, t in sorted(items)[:25]:
    print(f"  {k}: {t}")

# 3. Try Faithfulness class directly from submodule
print("\n=== Faithfulness class from submodule ===")
FaithClass = fm.Faithfulness
print(f"FaithClass: {FaithClass}")
print(f"issubclass(FaithClass, Metric): {issubclass(FaithClass, Metric)}")
print(f"MRO: {[c.__name__ for c in FaithClass.__mro__]}")

# 4. Instantiate with llm
print("\n=== Instantiate Faithfulness(llm=...) ===")
from openai import OpenAI
from ragas.llms import llm_factory
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-fake"))
llm = llm_factory("gpt-4o-mini", client=client)
print(f"llm type: {type(llm)}")
try:
    m = FaithClass(llm=llm)
    print(f"instance type: {type(m)}")
    print(f"isinstance(m, Metric): {isinstance(m, Metric)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# 5. Try fm.metric with llm set + evaluate
print("\n=== fm.metric with evaluate ===")
try:
    fm.metric.llm = llm
    print(f"isinstance(fm.metric, Metric): {isinstance(fm.metric, Metric)}")
    from ragas import evaluate
    from datasets import Dataset
    tiny = Dataset.from_dict({
        "question": ["test?"],
        "answer": ["test."],
        "contexts": [["test context"]],
        "ground_truth": ["test."],
    })
    print("Trying evaluate with fm.metric...")
    r = evaluate(tiny, metrics=[fm.metric], llm=llm)
    print(f"SUCCESS: {r}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
