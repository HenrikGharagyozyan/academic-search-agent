from app.providers.text_cleanup import strip_evidence_ids

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
