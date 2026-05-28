DISEASE_TASKS = {
    "Diabetes": {
        "slug": "diabetes",
        "target": "diabetes_present",
        "condition_patterns": ["diabetes", "diabetic"],
        "exclude_patterns": ["prediabetes"],
        "lab_rules": [("hba1c", ">=", 6.5), ("glucose", ">=", 200)],
    },
    "Hypertension": {
        "slug": "hypertension",
        "target": "hypertension_present",
        "condition_patterns": ["hypertension", "high blood pressure"],
        "exclude_patterns": [],
        "lab_rules": [("systolic_bp", ">=", 140), ("diastolic_bp", ">=", 90)],
    },
    "Obesity": {
        "slug": "obesity",
        "target": "obesity_present",
        "condition_patterns": ["obesity"],
        "exclude_patterns": [],
        "lab_rules": [("bmi", ">=", 30)],
    },
    "Heart Disease": {
        "slug": "heart_disease",
        "target": "heart_disease_present",
        "condition_patterns": [
            "coronary",
            "myocardial",
            "heart failure",
            "cardiac",
            "atrial fibrillation",
            "stroke",
        ],
        "exclude_patterns": [],
        "lab_rules": [],
    },
}


def disease_by_slug(slug: str) -> tuple[str, dict]:
    for name, task in DISEASE_TASKS.items():
        if task["slug"] == slug:
            return name, task
    raise KeyError(slug)


def all_targets() -> list[str]:
    return [task["target"] for task in DISEASE_TASKS.values()]
