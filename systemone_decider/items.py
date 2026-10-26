"""What a judge reads: one typed question, its options and a state, built exactly as in training.

Two settings:

    pair    one decision. Qwen2.5-1.5B-Instruct answers from nano-jev's top 2 HotpotQA paragraphs, Qwen2.5-7B-Instruct from
            all 10. The judge asks "Which model should answer this question?" {small model, large model}; option 1 is
            P(large model).
    menu    two decisions, weak first. Retrieval: Qwen2.5-7B selects passages, Qwen2.5-72B is the strong retriever.
            Answer: Llama-3.2-3B, with Llama-3.1-70B as the strong answerer. One model answers two yes/no questions;
            option 0 ("yes") is P(the weak stage suffices).

Inputs: `qp` reads the question and the passages; `qpa` also reads the weak model's answer.
"""

INPUTS = ("qp", "qpa")

PAIR = ("Which model should answer this question: {question}", ("small model", "large model"))

MENU = {
    "ret": "Do these passages hold what is needed to answer this question: {question}",
    "qp": "Will a small model answer this question correctly from these passages: {question}",
    "qpa": "Is the proposed answer correct? Question: {question}",
}
MENU_OPTIONS = ("yes", "no")  # option 0: the weak stage suffices


def check_inputs(inputs: str) -> None:
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {INPUTS}, not {inputs!r}")


def clean_answer(text: str) -> str:
    """A model's answer as the judge reads it in the menu setting: its first line, without a final full stop."""
    return text.strip().split("\n")[0].strip().rstrip(".")


def passages_state(paragraphs: list[str]) -> str:
    """Several "title: text" paragraphs as one state, numbered from 1."""
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def selection(text: str) -> str:
    """A retriever's output without its "Query:" line: what follows its first "Selected references:" (all of it if
    there is none), the selected references, "[k] title: text" per line. Training cut each output once, so pass it
    uncut: an output that repeats "Selected references:" reads differently cut twice."""
    return text.split("Selected references:", 1)[-1].strip() or "(no passages selected)"


def pair_item(question: str, passages: list[str], inputs: str, answer: str | None = None) -> dict:
    """The pair setting's item. `passages` are the small model's "title: text" paragraphs in the order it read them;
    `answer` (qpa) is the small model's answer as stored, not cleaned."""
    check_inputs(inputs)
    state = passages_state(passages)
    if inputs == "qpa":
        if answer is None:
            raise ValueError("the qpa judge reads the small model's answer: pass answer=")
        state = f"Proposed answer: {answer}\n{state}"
    return {"question": PAIR[0].format(question=question), "options": PAIR[1], "state": state}


def retrieval_item(question: str, weak_selection: str) -> dict:
    """The menu setting's first decision: do the weak retriever's selected passages hold what the answer needs?"""
    return {"question": MENU["ret"].format(question=question), "options": MENU_OPTIONS,
            "state": selection(weak_selection)}


def answer_item(question: str, branch_selection: str, inputs: str, weak_answer: str | None = None) -> dict:
    """The menu setting's second decision, under the retrieval that was chosen: will the weak answer do? qpa reads the
    weak answerer's answer on that branch's passages (its first line, at most 200 characters)."""
    check_inputs(inputs)
    state = selection(branch_selection)
    if inputs == "qpa":
        if weak_answer is None:
            raise ValueError("the qpa judge reads the weak model's answer: pass weak_answer=")
        state = f"Proposed answer: {clean_answer(weak_answer)[:200]}\n{state}"
    return {"question": MENU[inputs].format(question=question), "options": MENU_OPTIONS, "state": state}
