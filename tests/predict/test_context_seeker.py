import pytest
import dspy
from dspy import ContextSeeker, ContextSeekerTrainer, Oracle, ChainOfThought
from dspy.utils import DummyLM



@pytest.fixture
def dummy_lm():
    # 1. Stopping agent → ready=False
    # 2. Query agent → follow-up question
    # 3. Oracle → answer
    # 4. Stopping agent → ready=True
    # 5. Answer agent → final answer
    return DummyLM([
        {"ready": False},
        {"follow-up question": "What is the volume of the oceans?"},
        {"answer": "1.3e9"},
        {"ready": True},
        {"answer": "3.2e13"}
    ])


@pytest.fixture
def basic_example():
    return dspy.Example(
        question="How many golf balls would it take to submerge all land?",
        privileged_context="F1: Volume of a golf ball is 4e-5 km^3\nF2: Ocean volume is 1.3e9 km^3",
        answer="3.2e13"
    ).with_inputs("question", "privileged_context")


def test_context_seeker_initialization(dummy_lm):
    dspy.settings.configure(lm=dummy_lm)
    seeker = ContextSeeker(signature="question -> answer: str")
    assert seeker.query_agent is not None
    assert seeker.stopping_agent is not None
    assert seeker.answer_agent is not None


def test_context_seeker_forward(dummy_lm, basic_example):
    dspy.settings.configure(lm=dummy_lm)

    # Use the default Oracle chain-of-thought module
    oracle = Oracle()
    seeker = ContextSeeker(
        signature="question -> answer: str",
        oracle_module=oracle
    )
    seeker.set_privileged_context(basic_example.privileged_context)
    output = seeker(question=basic_example.question)

    assert isinstance(output.answer, str)
    assert output.answer == basic_example.answer  # optional: check answer match


def test_context_seeker_trainer(dummy_lm, basic_example):
    dspy.settings.configure(lm=dummy_lm)

    trainer = ContextSeekerTrainer(signature="question -> answer: str")
    pred = trainer(
        privileged_context=basic_example.privileged_context,
        question=basic_example.question
    )
    assert isinstance(pred.answer, str)
    assert pred.answer == basic_example.answer  # optional
