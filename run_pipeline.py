import subprocess

steps = [
    ("📅 Fetching upcoming fixtures", "fetch.fetch_fixtures"),
    ("🧱 Preparing training data", "model.prepare_training_data"),
    ("🧠 Training model", "model.train_model"),
    ("🎯 Generating predictions", "predict.predict_fixtures"),
]

def run_step(description, module_name):
    print(f"\n{description}...")
    result = subprocess.run(["python", "-m", module_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)

if __name__ == "__main__":
    print("🚀 Running Underdogged prediction pipeline...\n")
    for description, script in steps:
        run_step(description, script)
    print("\n✅ All steps completed!")
