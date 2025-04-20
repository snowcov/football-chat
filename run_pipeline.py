import os
import subprocess

def run_script(script_path, description):
    print(f"Running: {description} ({script_path})")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{description} completed successfully!")
        print(result.stdout)
    else:
        print(f"{description} failed!")
        print(result.stderr)
        exit(1)

if __name__ == "__main__":
    # Step 1: Train the tokenizer
    run_script("tokenizer.py", "Training the tokenizer")

    # Step 2: Save the tokenizer in Hugging Face format
    run_script("save_tokenizer.py", "Saving the tokenizer in Hugging Face format")

    # Step 3: Prepare and tokenize the dataset
    run_script("data_prep.py", "Preparing and tokenizing the dataset")

    # Step 4: Train the model
    run_script("train_model.py", "Training the model")

    # Step 5: Evaluate the model
    run_script("eval.py", "Evaluating the model")

    print("All steps completed successfully!")