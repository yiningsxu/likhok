from clinical_trial_qa.models import QuestionType, Reducer
from clinical_trial_qa.questions import all_question_specs, get_question_spec


def test_registry_has_15_boolean_and_8_numeric_specs():
    """Would fail if a criterion is omitted or assigned the wrong type/reducer."""
    specs = all_question_specs()

    assert len(specs) == 23
    assert sum(spec.question_type is QuestionType.BOOLEAN for spec in specs) == 15
    assert sum(spec.question_type is QuestionType.NUMERIC for spec in specs) == 8
    assert get_question_spec("PLT").reducer is Reducer.MIN
    assert get_question_spec("BILI").reducer is Reducer.MAX


def test_lvef_spec_carries_legacy_output_cap():
    """Would fail if compatibility clipping were left only in prompt prose."""
    spec = get_question_spec("lvef")

    assert spec.output_floor_or_cap == 55.0
    assert spec.reducer is Reducer.MIN
