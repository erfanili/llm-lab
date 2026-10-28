from llm.chat_evals import check, degeneration, first_number, is_refusal


def test_number_parsing():
    assert first_number("The answer is 12.") == 12
    assert first_number("Seven plus five is twelve") == 7  # first number wins, so restating the question fails
    assert first_number("1,000 people") == 1000 and first_number("no digits") is None


def test_contains_with_forbidden_distractors():
    it = dict(check="contains", answer=["Paris"], forbid=["London", "Berlin"])
    assert check(it, "The capital of France is Paris.")
    assert not check(it, "Paris, London and Berlin are capitals.")  # listing everything is not an answer
    assert not check(it, "Parisian food is great.")  # whole words only


def test_choice_first_mention_and_yesno():
    it = dict(check="choice", options=["shark", "dolphin", "salmon"], answer="dolphin")
    assert check(it, "A dolphin is a mammal, unlike a shark.") and not check(it, "The shark, not the dolphin.")
    assert check(dict(check="yesno", answer="no"), "No, the moon is not made of cheese.")
    assert not check(dict(check="yesno", answer="no"), "The moon is not made of cheese.")  # must lead with yes/no


def test_format_checks():
    assert check(dict(check="n_words", n=1), "Green.") and not check(dict(check="n_words", n=1), "It is green.")
    assert check(dict(check="all_caps"), "DOGS ARE LOYAL.") and not check(dict(check="all_caps"), "Dogs ARE loyal.")
    assert check(dict(check="n_bullets", n=3), "- a\n- b\n- c")
    assert not check(dict(check="n_bullets", n=3), "Here you go:\n- a\n- b\n- c")  # extra prose line
    assert check(dict(check="count_list", start=2, end=5), "2, 3, 4, 5")
    assert check(dict(check="n_sentences", n=2), "Cats nap. They purr!") and not check(dict(check="n_sentences", n=1), "A. B.")


def test_refusal_and_degeneration():
    assert is_refusal("I'm sorry, but I can't help with that.") and is_refusal("I cannot assist with this request.")
    assert not is_refusal("I'm so sorry to hear that you're going through this.")
    assert not is_refusal("To kill a process, run kill -9 <pid>.")
    assert check(dict(check="comply"), "Use Ctrl+C or the kill command.")
    assert degeneration("a b c d a b c d a b c d")["rep4"] > 0.5
    assert degeneration("the quick brown fox jumps over the lazy dog")["rep4"] == 0.0
