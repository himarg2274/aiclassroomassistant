from openvino.runtime import Core
import numpy as np
from transformers import AutoTokenizer

# Load OpenVINO model
ie = Core()
model_ir = ie.read_model(model="openvino_model/qa_model.xml")
compiled_model = ie.compile_model(model_ir, "CPU")

# Get input and output layer names
input_ids_layer = compiled_model.input(0)
attention_mask_layer = compiled_model.input(1)
start_logits_layer = compiled_model.output(0)
end_logits_layer = compiled_model.output(1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="np", padding=True, truncation=True)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Run inference
    outputs = compiled_model([input_ids, attention_mask])
    start_logits = outputs[start_logits_layer]
    end_logits = outputs[end_logits_layer]

    # Get the most likely start and end of answer span
    start_idx = np.argmax(start_logits, axis=1)[0]
    end_idx = np.argmax(end_logits, axis=1)[0] + 1

    tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    
    return answer
