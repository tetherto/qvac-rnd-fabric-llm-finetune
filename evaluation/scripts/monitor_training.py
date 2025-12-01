#!/usr/bin/env python3
import os
import time
import glob
from pathlib import Path

def monitor_training():
    print("🔍 Monitoring biomedical yes/no training...")
    
    # Find the latest run directory
    run_dirs = glob.glob("runs/biomed_yesno_*")
    if not run_dirs:
        print("❌ No training runs found")
        return
    
    latest_run = max(run_dirs, key=os.path.getctime)
    print(f"📁 Latest run: {latest_run}")
    
    eval_csv = Path(latest_run) / "eval_progress.csv"
    
    while True:
        # Check if process is still running
        import subprocess
        result = subprocess.run(["pgrep", "-f", "train_biomed_yesno"], 
                              capture_output=True, text=True)
        
        if not result.stdout.strip():
            print("🏁 Training process has finished!")
            break
            
        # Check for eval progress
        if eval_csv.exists():
            try:
                with open(eval_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Skip header
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split(',')
                            if len(parts) >= 3:
                                step, acc, f1 = parts[0], parts[1], parts[2]
                                print(f"📊 Step {step}: Accuracy={acc}, Macro-F1={f1}")
            except:
                pass
        else:
            print("⏳ Still loading model... (this can take 5-10 minutes on CPU)")
        
        time.sleep(30)  # Check every 30 seconds
    
    # Show final results if available
    if eval_csv.exists():
        print("\n📈 Final training progress:")
        try:
            import pandas as pd
            df = pd.read_csv(eval_csv)
            print(df.to_string(index=False))
        except:
            with open(eval_csv, 'r') as f:
                print(f.read())
    
    # Check for test results
    test_results = Path(latest_run) / "final_test_results.csv"
    if test_results.exists():
        print(f"\n✅ Test results saved to: {test_results}")
        try:
            import pandas as pd
            df = pd.read_csv(test_results)
            accuracy = df['correct'].mean()
            print(f"🎯 Final test accuracy: {accuracy:.3f}")
            print(f"📝 Sample predictions:")
            print(df[['question', 'gold_label', 'predicted_label', 'correct']].head())
        except Exception as e:
            print(f"Could not parse results: {e}")

if __name__ == "__main__":
    monitor_training()

