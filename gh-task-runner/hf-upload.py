import os
import sys
import time
from huggingface_hub import HfApi

# Get the arguments, passed to the python script
args = sys.argv

# Check if the 3 args are set
# show an error, and example usage if not
if len(args) != 4:
    print(f"# ------------------------------------")
    print(f"# Error: Missing arguments")
    print(f"# ------------------------------------")
    print(f"# Usage: python hf-upload.py <REPO_URI> <NOTEBOOK_SUBDIR> <OUTPUT_DIR>")
    print(f"# Example: python hf-upload.py username/project-name notebook-name output")
    print(f"# ------------------------------------")
    sys.exit(1)

# Build the UPLOAD_SUBDIR, from the first argument
HF_UPLOAD_REPO = args[1]
HF_UPLOAD_SUBDIR = args[2]
UPLOAD_DIR = args[3]

# Remove ending / from the HF_UPLOAD_SUBDIR
HF_UPLOAD_SUBDIR = HF_UPLOAD_SUBDIR.rstrip('/')

# This hopefully fix some issues with the HF API
import os
os.environ['CURL_CA_BUNDLE'] = ''

# Get the Hugging Face Hub API
api = HfApi()

# Directory paths
RUNNER_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(RUNNER_SCRIPT_DIR)

# Generate the URL where all the items will be uploaded
hf_url = f"https://huggingface.co/datasets/{HF_UPLOAD_REPO}/tree/main/{HF_UPLOAD_SUBDIR}"
print(f"# ------------------------------------")
print(f"# Uploading to: {hf_url}")
print(f"# ------------------------------------")

# ------------------------------------
# Uploading code, with work around some unstable HF stuff
# ------------------------------------

# List of errors, to throw at the end of all upload attempts
UPLOAD_ERRORS = []

# Fallback upload method, upload the files one by one
# with retry on failure (up to 3 attempts)
def upload_folder_fallback(folder_path, file_types=["json", "jsonl", "log", "txt"], log_type="various"): 
    # Get the files to upload in the folder_path, including nested files
    file_list = []

    # Walk the folder and get all the files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    # Filter the file_list by file_types
    file_list = [f for f in file_list if f.split('.')[-1] in file_types]

    # Log the fallback logic
    print(f"# Fallback {log_type} upload method ... ")

    # Upload the files one by one
    for file in file_list:

        file_abs = os.path.join(folder_path, file)
        file_rel = os.path.relpath(file_abs, folder_path)
        print(f"# Uploading {log_type} file: {file_rel} ... ")

        for i in range(3):
            try:
                api.upload_file(
                    path_or_fileobj=file_abs,
                    repo_id=HF_UPLOAD_REPO,
                    path_in_repo=f"{HF_UPLOAD_SUBDIR}/{file_rel}",
                    repo_type="dataset",
                    commit_message=f"[GHA] {HF_UPLOAD_SUBDIR}/{file_rel} (fallback single file upload)"
                )
            except Exception as e:
                print(f"# Error uploading {log_type} file: {file_rel} ... ")
                print(e)
                if i == 2:
                    UPLOAD_ERRORS.append(e)
                # Minor sleep before retry
                time.sleep(5)
                continue
            break
    
    # Upload finished !
    print(f"# Upload of {log_type} files, via fallback finished !")

# Because multi-stage upload is "unstable", we try to upload the file with fallback handling
def upload_folder(folder_path, log_type="various"): 
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=HF_UPLOAD_REPO,
            path_in_repo=HF_UPLOAD_SUBDIR,
            repo_type="dataset",
            multi_commits=True,
            allow_patterns=["*.json", "*.jsonl", "*.txt", "*.log","*/*.json", "*/*.jsonl"],
            commit_message=f"[GHA] {HF_UPLOAD_SUBDIR} - {log_type}"
        )
    except Exception as e:
        eStr = str(e)
        if "must have at least 1 commit" in eStr:
            print("# Skipping {file_type} upload due to error ... ")
            print(e)
        else:
            upload_folder_fallback(folder_path, ["json", "jsonl", "log", "txt"], log_type)

# ------------------------------------
# Actual upload (and error handling)
# ------------------------------------

# Upload the ipynb files
print("# Uploading the various files ... ")
upload_folder( f"{UPLOAD_DIR}", log_type="various" )

print(f"# ------------------------------------")
print(f"# Uploaded finished to: {hf_url}")
print(f"# ------------------------------------")

# Print out the errors, if any
if len(UPLOAD_ERRORS) > 0:
    print("# ------------------------------------")
    print("# Upload errors:")
    print("# ------------------------------------")
    for e in UPLOAD_ERRORS:
        print(e)
    print("# ------------------------------------")
    print("# Upload errors, logged - hard exit")
    print("# ------------------------------------")
    sys.exit(1)