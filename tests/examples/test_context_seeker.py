"""
Example: ContextSeeker Module on Fermi-Style Estimation QA

This script demonstrates how to use the ContextSeekerTrainer to handle underspecified
questions by asking follow-up queries before producing a final answer.
"""

import dspy
from dspy.predict.context_seeker import ContextSeekerTrainer
import json

# Setup the language model
lm = dspy.LM(
    "openai/google/gemma-3-12b-it", 
    api_base="http://localhost:8000/v1", 
    api_key="EMPTY", 
    model_type="chat", 
    temperature=0.1,
    max_tokens=8000
)
dspy.configure(lm=lm)


def load_local_fermi_json(filepath: str, split_ratio: float = 0.7):
    """
    Loads the locally stored Fermi-style JSON dataset and converts it into dspy.Examples.

    Args:
        filepath (str): Path to the local JSON file.
        split_ratio (float): Train/val split ratio. Default is 0.7.

    Returns:
        trainset, valset (List[dspy.Example], List[dspy.Example])
    """
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    examples = []
    for ex in raw_data:
        question = ex.get("question")
        context = ex.get("context")
        answer = ex.get("answer") or str(ex.get("numerical_answer"))
        unit = ex.get('unit') or 'count'
        allowed_questions = ex.get('variables') or 'open ended'

        if not question or not context or not answer:
            continue  # skip incomplete rows

        context = context.replace("CONTEXT:=", "").strip()

        examples.append(
            dspy.Example(
                question=question,
                unit=unit,
                allowed_questions=allowed_questions,
                privileged_context=context,
                answer=answer
            ).with_inputs("question", "unit", "allowed_questions", "privileged_context")
        )

    split = int(split_ratio * len(examples))
    return examples[:split], examples[split:]

# Load dataset
trainset, valset = load_local_fermi_json("tests/fermi_formatted.json") # TODO: might have to change this path

# Define custom accuracy metric
import numpy as np
def log_order_accuracy(ex, pred, trace=None):
    try:
        agent_answer = float(pred.answer)
        correct_answer = float(ex.answer.strip())
        if agent_answer == 0 and correct_answer == 0:
            return 1.0
        if agent_answer == 0 or correct_answer == 0:
            return 0.0
        order_diff = abs(np.log10(agent_answer / correct_answer))
        return max(0, 1 - order_diff / 3)
    except:
        return 0.0

# Initialize trainer
trainer = ContextSeekerTrainer(
    signature="question, unit, allowed_questions -> answer: float",
    includes_allowed_questions=True
)

# Evaluate baseline
evaluate = dspy.Evaluate(devset=valset, metric=log_order_accuracy, display_progress=True, display_table=5)
evaluate(trainer)

# Optimize with MIPRO
optimizer = dspy.MIPROv2(metric=log_order_accuracy, num_threads=64, teacher_settings=dict(lm=lm), auto="medium")
optimized_trainer = optimizer.compile(trainer, trainset=trainset)

# Final eval
evaluate(optimized_trainer)

# Optional: switch to human oracle for debugging
def human_oracle(question, privileged_context=None):
    print("Follow-up:", question)
    if privileged_context:
        print("Privileged context:", privileged_context)
    return dspy.Prediction({"answer": input("Your answer: ")})

trainer.update_oracle(human_oracle)
output = trainer(**valset[0].inputs())
print("Answer:", output.answer)
