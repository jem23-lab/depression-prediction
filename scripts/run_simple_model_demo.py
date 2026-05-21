import importlib.util
from pathlib import Path
import sys


def _load_model_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    module_path = repo_root / "shared" / "depression_model copy.py"
    spec = importlib.util.spec_from_file_location("depression_model_copy", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    model = _load_model_module()
    model.load()

    samples = [
        (
            "severe_example",
            "I feel drained almost every day, both physically and mentally. It is hard to get out of bed, "
            "and I often skip responsibilities because I cannot find the energy. I avoid people and feel like "
            "I do not belong anywhere. My thoughts are mostly negative, and I feel stuck in this state.",
        ),
        (
            "moderate_example",
            "I have been feeling down and a bit tired, with some trouble sleeping. I still enjoy a few "
            "things, but it is harder to stay focused and I feel a little hopeless at times.",
        ),
        (
            "not_depressed_example",
            "Overall I feel okay. I stay active, sleep fairly well, and enjoy my work, even though I have "
            "some everyday stress.",
        ),
    ]

    for sample_id, text in samples:
        probs = model.predict_proba([text])[0]
        label, score, reason = model.classify_severity(probs, text=text)
        print(f"Sample ID: {sample_id}")
        print(f"Text: {text}")
        print(f"Probs: {probs}")
        print(f"Label: {label}")
        print(f"Score: {score}")
        print(f"Reason: {reason}")
        print("-")


if __name__ == "__main__":
    main()
