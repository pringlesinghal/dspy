import dspy
from dspy import ContextSeeker, ContextSeekerTrainer, ChainOfThought
from dspy.utils import DummyLM
import pytest


@pytest.fixture
def dummy_lm():
    lm = DummyLM([
        {"reasoning": "Need to check ocean volume", "ready": True},
        {"reasoning": "Need to check ocean volume", "follow-up question": "What is the volume of the oceans?"},
        {"reasoning": "Found ocean volume", "answer": "1.3e9"},
        {"reasoning": "Ready to calculate", "ready": True},
        {"reasoning": "Calculated final answer", "answer": "3.2e13"}
])
    dspy.settings.configure(lm=lm)
    return lm

@pytest.fixture
def basic_example():
    return dspy.Example(
        question="How many golf balls would it take to submerge all land?",
        privileged_context="F1: Volume of a golf ball is 4e-5 km^3\nF2: Ocean volume is 1.3e9 km^3",
        answer="3.2e13"
    ).with_inputs("question", "privileged_context")


def test_context_seeker_initialization(dummy_lm, caplog):
    seeker = ContextSeeker(signature="question -> answer: str")
    # caplog.set_level("INFO")
    caplog.messages(f"Query agent type: {type(seeker.query_agent)}")
    caplog.messages(f"Stopping agent type: {type(seeker.stopping_agent)}")
    caplog.messages(f"Answer agent type: {type(seeker.answer_agent)}")

    assert seeker.query_agent is not None
    assert seeker.stopping_agent is not None
    assert seeker.answer_agent is not None

