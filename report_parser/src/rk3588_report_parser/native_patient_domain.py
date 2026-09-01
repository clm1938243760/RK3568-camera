from __future__ import annotations

from typing import Any


STANDARD_PATIENT_FIELDS = (
    "birthday",
    "exam_item",
    "ming",
    "sex",
    "yue",
    "his_exam_no",
    "xing",
    "patient_id",
    "ri",
    "patient_name",
    "name_phonetic",
    "nian",
    "report_no",
    "age",
)


class ValidationError(RuntimeError):
    pass


def canonical_patient(record: dict[str, Any]) -> dict[str, Any]:
    patient = {field: record.get(field) for field in STANDARD_PATIENT_FIELDS}
    extra_fields = record.get("extra_fields", {})
    patient["extra_fields"] = dict(extra_fields) if isinstance(extra_fields, dict) else {}
    return patient
