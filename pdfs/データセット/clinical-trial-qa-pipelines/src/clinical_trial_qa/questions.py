"""Canonical registry for the 23 approved trial criteria."""

from __future__ import annotations

from .models import QuestionSpec, QuestionType, Reducer


def _boolean(
    criterion: str,
    question: str,
    aliases: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    time_condition: str | None = None,
    subject_condition: str | None = "patient",
) -> QuestionSpec:
    return QuestionSpec(
        criterion=criterion,
        question_type=QuestionType.BOOLEAN,
        question=question,
        aliases=aliases,
        labels=labels,
        time_condition=time_condition,
        subject_condition=subject_condition,
    )


def _numeric(
    criterion: str,
    question: str,
    aliases: tuple[str, ...],
    expected_unit: str | None,
    reducer: Reducer,
    *,
    output_floor_or_cap: float | None = None,
) -> QuestionSpec:
    return QuestionSpec(
        criterion=criterion,
        question_type=QuestionType.NUMERIC,
        question=question,
        aliases=aliases,
        expected_unit=expected_unit,
        labels=("laboratory",),
        reducer=reducer,
        output_floor_or_cap=output_floor_or_cap,
    )


_SPECS = (
    _boolean("afib", "Does the note describe the patient as having permanent, paroxysmal, or persistent afib?", ("afib", "atrial fibrillation"), ("diagnosis", "cardiology")),
    _boolean("afib_ablation", "Does the note describe the patient as having a planned or past ablation procedure for afib?", ("ablation", "pulmonary vein isolation"), ("procedure", "cardiology")),
    _boolean("arterial_hypertension", "Does the note describe the patient as having arterial hypertension on treatment (high bp e.g. >140, or HTN)?", ("hypertension", "HTN", "high blood pressure"), ("diagnosis", "history", "medication")),
    _boolean("bipolar", "Does the note describe the patient as ever being diagnosed with bipolar disorder?", ("bipolar", "manic depressive"), ("diagnosis", "psychiatry"), time_condition="ever"),
    _boolean("bleeding", "Does the note describe the patient as having a serious bleeding in the past 6 months?", ("bleeding", "hemorrhage", "GI bleed"), ("bleeding", "history"), time_condition="past 6 months"),
    _numeric("blood_glucose", "What is the highest blood glucose lab mentioned? Answer \"NA\" if no blood glucose score is in the note.", ("glucose", "blood glucose", "BG"), "mg/dL", Reducer.MAX),
    _numeric("chads2", "What is the highest CHADS2 score mentioned? Answer \"NA\" if no CHADS2 score is in the note. ", ("CHADS2", "CHADS-2"), None, Reducer.MAX),
    _boolean("heart_failure", "Does the note describe the patient as having heart failure?", ("heart failure", "CHF", "congestive heart failure"), ("diagnosis", "history", "cardiology")),
    _boolean("hemorrhagic", "Does the note describe the patient as ever having any hemorrhagic tendencies or blood dyscrasias?", ("hemorrhagic", "blood dyscrasia", "bleeding tendency"), ("bleeding", "history"), time_condition="ever"),
    _numeric("lvef", "What is the lowest left ventricular ejection (LVEF, ef, ejection fraction) fraction mentioned in the note? Answer \"NA\" if no LVEF is in the note, Answer 55 if the lowest value is 55% or greater.", ("LVEF", "EF", "ejection fraction"), "%", Reducer.MIN, output_floor_or_cap=55.0),
    _boolean("mdd", "Does the note describe the patient as ever being diagnosed with depression or major depressive disorder (MDD)?", ("depression", "major depressive disorder", "MDD"), ("diagnosis", "psychiatry"), time_condition="ever"),
    _boolean("med_decisions", "Does the note describe the patient as being unable to make medical decisions? (Answer no unless there is evidence the patient cannot make their own medical decisions).", ("medical decisions", "decision-making capacity", "incapacitated"), ("capacity", "psychiatry")),
    _boolean("peptic_ulcer_disease", "Does the note describe the patient as ever having peptic ulcer disease?", ("peptic ulcer", "PUD", "gastric ulcer"), ("diagnosis", "history", "bleeding"), time_condition="ever"),
    _boolean("prior_stroke", "Does the note describe the patient as ever having a stroke or transient ischemic attack?", ("stroke", "TIA", "transient ischemic attack"), ("diagnosis", "history", "neurology"), time_condition="ever"),
    _boolean("recent_stroke", "Does the note describe the patient as having a stroke during this admission or within the last month? (Answer yes for any recent stroke if the date is unclear)", ("stroke", "TIA", "transient ischemic attack"), ("diagnosis", "history", "neurology"), time_condition="this admission or past month"),
    _boolean("schizophrenia", "Does the note describe the patient as ever being diagnosed with schizophrenia or any schizoaffective disorders?", ("schizophrenia", "schizoaffective"), ("diagnosis", "psychiatry"), time_condition="ever"),
    _boolean("surgical_valvular_disease", "Does the note describe the patient as ever having valvular disease requiring surgery (stenosis)?", ("valvular", "valve surgery", "stenosis"), ("diagnosis", "procedure", "cardiology"), time_condition="ever"),
    _boolean("t2d", "Does the note describe the patient as having Diabetes mellitus (DM1, DM2, T2D, T1DM, T2DM)?", ("diabetes", "DM1", "DM2", "T1DM", "T2DM", "T2D"), ("diagnosis", "history")),
    _numeric("CREAT", "What is the higest serum creatinine (Creat) mentioned in the note? Answer \"NA\" if no creatinine value is available in the note.", ("creatinine", "Creat", "Cr"), "mg/dL", Reducer.MAX),
    _numeric("AST", "What is the higest aspartate aminotransferase level (AST) mentioned in the note? Answer \"NA\" if no AST value is available in the note.", ("AST", "aspartate aminotransferase"), "U/L", Reducer.MAX),
    _numeric("BILI", "What is the higest total bilirubin (TotBili, Bili) mentioned in the note? Answer \"NA\" if no bilirubin value is available in the note.", ("bilirubin", "TotBili", "Bili"), "mg/dL", Reducer.MAX),
    _numeric("PLT", "What is the lowest platelet count (PLT) mentioned in the note? Answer \"NA\" if no platelet count is available in the note.", ("platelet", "PLT"), "K/uL", Reducer.MIN),
    _numeric("HGB", "What is the lowest hemoglobin (HGB) mentioned in the note? Answer \"NA\" if no HGB value is available in the note.", ("hemoglobin", "HGB", "Hb"), "g/dL", Reducer.MIN),
)

_BY_CRITERION = {spec.criterion: spec for spec in _SPECS}


def all_question_specs() -> tuple[QuestionSpec, ...]:
    return _SPECS


def get_question_spec(criterion: str) -> QuestionSpec:
    try:
        return _BY_CRITERION[criterion]
    except KeyError as exc:
        raise KeyError(f"unknown criterion: {criterion}") from exc
