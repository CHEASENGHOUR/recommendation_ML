import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from pipelines.training_pipeline import run_training_pipeline


if __name__ == "__main__":
    # ── Validate data file ─────────────────────────────────────────────
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/laptop_data.csv"

    if not os.path.exists(data_path):
        print(f"   Data file not found: {data_path}")
        print("    Place your CSV at data/laptop_data.csv  OR pass the path as an argument:")
        print("    python main.py path/to/your_data.csv")
        sys.exit(1)

    # ── Run full training pipeline ─────────────────────────────────────
    version, model_path = run_training_pipeline(data_path)

    print(f"\n Next steps:")
    print(f"   1. Start Django API:  python manage.py runserver")
    print(f"   2. Test endpoint:     curl http://127.0.0.1:8000/api/recommend/?q=gaming+laptop")