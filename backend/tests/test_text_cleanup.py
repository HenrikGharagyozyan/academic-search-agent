from app.domain.text.cleanup import find_evidence_id_leak, strip_evidence_ids

# Strings copied from answers the model actually produced.
A = "135fb597-524d-4b09-8b9c-1516cb978b-444a-b2c-934cb8af1918"
B = "005fe97b-b07a-45e4-881b-d6d829c6ab97"
C = "dc95352e-d549-415c-9974-1689e3cc3bea"


def test_removes_trailing_parenthetical_with_one_id():
    text = f"This is useful for data-processing tasks (evidence_ids: {B})."
    assert strip_evidence_ids(text) == "This is useful for data-processing tasks."


def test_removes_trailing_parenthetical_with_several_ids():
    text = f"Outputs are validated according to user needs (evidence_ids: {B}, {C})."
    assert strip_evidence_ids(text) == "Outputs are validated according to user needs."


def test_removes_bracketed_form():
    text = f"The method supports include_raw [evidence_id: {B}] for debugging."
    assert strip_evidence_ids(text) == "The method supports include_raw for debugging."


def test_collapses_bare_ids_joined_by_and():
    text = (
        f"Evidence supports this through descriptions in both evidence_id: {B} "
        f"and evidence_id: {C}, indicating consensus."
    )
    assert strip_evidence_ids(text) == (
        "Evidence supports this through descriptions in both, indicating consensus."
    )


def test_removes_unlabelled_id():
    text = f"The schema is validated ({B}) before use."
    assert strip_evidence_ids(text) == "The schema is validated before use."


def test_leaves_ordinary_prose_untouched():
    text = "The with_structured_output method returns a Pydantic instance."
    assert strip_evidence_ids(text) == text


def test_leaves_maths_and_punctuation_alone():
    text = r"The tensor $G_{\mu\nu}$ is symmetric, and $\Lambda$ is constant."
    assert strip_evidence_ids(text) == text


def test_keeps_word_boundary_when_bare_id_sits_mid_sentence():
    # The id patterns eat the whitespace on both sides, so a naive removal
    # ran the neighbouring words together ("Adam <id> converges" ->
    # "Adamconverges"). The surrounding prose must stay readable.
    text = f"Adam {B} converges quickly."
    assert strip_evidence_ids(text) == "Adam converges quickly."


def test_keeps_word_boundary_around_a_run_of_bare_ids():
    text = f"Two ids {B} {C} here."
    assert strip_evidence_ids(text) == "Two ids here."


def test_keeps_word_boundary_before_a_connector():
    text = f"Values {B} and cost matter."
    assert strip_evidence_ids(text) == "Values and cost matter."


# --- ids the model copied wrongly -------------------------------------------

# From a live answer: the first group is seven characters, not eight. The old
# pattern demanded an exact UUID and let this reach the reader.
MANGLED = "72cacfe-cb0a-435f-a3f4-7b7a3bd1bfc0"


def test_removes_bracketed_id_glued_to_the_full_stop():
    text = f"...to improve exponentially.[evidence_id: {MANGLED}]"
    assert strip_evidence_ids(text) == "...to improve exponentially."


def test_removes_a_well_formed_id_glued_to_the_full_stop():
    text = f"...to improve exponentially.[evidence_id: {B}]"
    assert strip_evidence_ids(text) == "...to improve exponentially."


def test_removes_mangled_id_in_every_position():
    assert strip_evidence_ids(f"Rates fall (evidence_ids: {MANGLED}, {B}).") == "Rates fall."
    assert strip_evidence_ids(f"Rates fall evidence_id: {MANGLED}.") == "Rates fall."
    assert strip_evidence_ids(f"Rates fall ({MANGLED}).") == "Rates fall."


def test_removes_two_ids_run_together():
    assert strip_evidence_ids(f"Rates fall [evidence_id: {A}].") == "Rates fall."


def test_removes_a_labelled_bracket_whatever_it_holds():
    assert strip_evidence_ids("Rates fall [evidence_id: chunk 3].") == "Rates fall."


def test_leaves_dates_and_versions_alone():
    text = "Released 2024-05-01 as v1.2.3-rc-1, the fix covers a-b-c-d-e."
    assert strip_evidence_ids(text) == text


# --- the last check before the reader ---------------------------------------


def test_finds_nothing_in_clean_text():
    assert find_evidence_id_leak("Logical error rates fall exponentially.") is None
    assert find_evidence_id_leak("The evidence identifies two regimes.") is None


def test_finds_a_leftover_label():
    assert find_evidence_id_leak("as shown in evidence_id 3") == "evidence_id"


def test_finds_a_leftover_id_even_inside_a_fence():
    text = f"```text\nqubit -> {MANGLED}\n```"
    assert find_evidence_id_leak(text) == MANGLED
