from openvino.runtime import Core
import numpy as np
from transformers import AutoTokenizer

from transformers import pipeline
qa_pipeline = pipeline("question-answering")

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']


    tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    
    return answer
