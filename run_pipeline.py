import subprocess

steps = [
    ("📅 Fetching upcoming fixtures", "fetch/fetch_fixtures.py"),
    ("💸 Fetching latest odds", "fetch/fetch_odds.py"),
    ("🧱 Preparing training data", "model/prepare_training_data.py"),
    ("🧠 Training model", "model/train_model.py"),
    ("🎯 Generating predictions", "predict/predict_fixtures.py"),
]

def run_step(description, script):
    print(f"\n{description}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)

if __name__ == "__main__":
    print("🚀 Running Underdogged prediction pipeline...\n")
    for description, script in steps:
        run_step(description, script)
    print("\n✅ All steps completed!")
