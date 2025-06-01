from typing import Optional, Type, Union, List, Tuple, Callable
from copy import deepcopy

import dspy
from dspy.primitives.program import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature, _default_instructions


# Oracle module that uses privileged context
class OracleModule(Module):
    def __init__(self):
        signature = Signature({
            "question": (str, InputField(prefix="Follow-up question:", desc="The question to answer", type=str)),
            "privileged_context": (str, InputField(prefix="Privileged context:", desc="Context that the oracle can use", type=str)),
            "answer": (str, OutputField(prefix="Answer:", desc="The answer to the question", type=str))
        })
        signature.instructions = _default_instructions(signature)
        self.oracle = dspy.ChainOfThought(signature=signature)

    def forward(self, question: str, privileged_context: str) -> dspy.Prediction:
        return self.oracle(question=question, privileged_context=privileged_context)


# ContextSeeker module
class ContextSeeker(Module):
    def __init__(
        self,
        signature: Union[str, Type[Signature]],
        stopping_module: Optional[Type[Module]] = dspy.ChainOfThought,
        query_module: Optional[Type[Module]] = dspy.ChainOfThought,
        answer_module: Optional[Type[Module]] = dspy.ChainOfThought,
        oracle_module: Optional[Module] = None,
        budget: int = 5,
    ):
        self.signature = ensure_signature(signature)
        self.follow_up_question_type = str
        self.follow_up_response_type = str
        self.context_type = List[Tuple[str, str]]
        self.budget = budget

        self.stopping_agent = stopping_module(signature=self._generate_signature_stopping_agent())
        self.query_agent = query_module(signature=self._generate_signature_query_agent())
        self.answer_agent = answer_module(signature=self._generate_signature_answer_agent())

        self.oracle_module = oracle_module
        self._current_privileged_context = None

    def set_privileged_context(self, privileged_context: str):
        self._current_privileged_context = privileged_context

    def _add_context_to_input_fields(self, input_fields):
        input_fields["context"] = (
            self.context_type,
            InputField(
                prefix="Context:",
                desc="List of (follow-up question, response) tuples.",
                type=self.context_type,
            ),
        )

    def _generate_signature_stopping_agent(self):
        input_fields = dict(self.signature.input_fields)
        self._add_context_to_input_fields(input_fields)
        input_fields["ready"] = (
            bool,
            OutputField(prefix="Ready:", desc="Whether to stop clarifying.", type=bool)
        )
        return Signature(input_fields, instructions=_default_instructions(Signature(input_fields)))

    def _generate_signature_query_agent(self):
        input_fields = dict(self.signature.input_fields)
        self._add_context_to_input_fields(input_fields)
        input_fields["follow-up question"] = (
            str,
            OutputField(prefix="Follow-up Question:", desc="Clarifying question.", type=str)
        )
        return Signature(input_fields, instructions=_default_instructions(Signature(input_fields)))

    def _generate_signature_answer_agent(self):
        input_fields = dict(self.signature.input_fields)
        self._add_context_to_input_fields(input_fields)
        input_fields.update(self.signature.output_fields)
        return Signature(input_fields, instructions=_default_instructions(Signature(input_fields)))

    def forward(self, **kwargs):
        context = []
        stopping_result = self.stopping_agent(**kwargs, context=context)
        num_questions = 0

        while not stopping_result.ready and num_questions < self.budget:
            query_result = self.query_agent(**kwargs, context=context)
            question = query_result["follow-up question"]


            oracle_result = self.oracle_module(question=question, privileged_context=self._current_privileged_context)
            response = oracle_result.answer
            context.append((question, response))
            stopping_result = self.stopping_agent(**kwargs, context=context)
            num_questions += 1

        return self.answer_agent(**kwargs, context=context)


# Trainer module to wrap ContextSeeker and handle privileged oracle access
class ContextSeekerTrainer(Module):
    def __init__(
        self,
        signature: Union[str, Type[Signature]],
        oracle_module: Type[Module] = OracleModule,
        stopping_module: Optional[Type[Module]] = dspy.ChainOfThought,
        query_module: Optional[Type[Module]] = dspy.ChainOfThought,
        answer_module: Optional[Type[Module]] = dspy.ChainOfThought,
        budget: int = 5,
    ):
        self.signature = ensure_signature(signature)
        self.oracle_model = oracle_module()
        self.context_seeker = ContextSeeker(
            signature=self.signature,
            oracle_module=self.oracle_model,
            stopping_module=stopping_module,
            query_module=query_module,
            answer_module=answer_module,
            budget=budget
        )

    def forward(self, privileged_context: str, **kwargs):
        self.context_seeker.set_privileged_context(privileged_context)
        return self.context_seeker(**kwargs)

    def extract_context_seeker(self) -> ContextSeeker:
        seeker = deepcopy(self.context_seeker)
        return seeker
    
    def update_oracle(self, new_oracle: Optional[Callable] = None):
        """
        Update the oracle used by the ContextSeeker.
        
        Args:
            new_oracle (Callable): A new oracle function or module to use.
        """
        if new_oracle is not None:
            self.context_seeker.oracle_module = new_oracle

