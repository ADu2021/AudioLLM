# /juice2/scr2/duyy/models/tts-scores
import os

from tts_scores.clvp import CLVPMetric
from tts_scores.intelligibility import IntelligibilityMetric

# BASE_PATH: output audio path
# BASE_PATH = "/sailhome/duyy/data/AudioLLM/TTU/ckpt-2024-08-15_11-18-01-LLaMA-UnitEmbed-CANINE_sum-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/nolast"
BASE_PATH = "/sailhome/duyy/data/AudioLLM/TTU/zdemo"
SPLIT = "val"
assert SPLIT in ["train", "val", "rotate_val"]

REF_PATH = "/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ckpt-2024-08-12_00-20-39-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-2_tgt_mask-lookbehind-2_mem_mask/answer"
CLVP_PATH = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/tts-scores-clvp/clvp.pth"

tsv_filename = f"{SPLIT}_50.tsv"

def extract_tsv(directory=BASE_PATH, split=SPLIT):
    # Specify the directory containing the files
    
    
    # List to store file data
    file_data = []

    # Open a TSV file to write the output
    with open(os.path.join(directory, tsv_filename), 'w', encoding='utf-8') as tsv_file:
        # Iterate over files in the specified directory
        for filename in os.listdir(directory):
            if filename.endswith('.wav') and '_' in filename:
                # Extract the integer ID from the filename
                id = int(filename.split('_', 1)[0])
                if (split == "train" and id < 50) or (split == "val" and id >= 50 and id < 100):
                    new_filename = f"{id}.wav"  # New filename based on the ID
                    new_file_path = os.path.join(directory, new_filename)
                    original_file_path = os.path.join(directory, filename)
                    
                    # Rename the file
                    os.rename(original_file_path, new_file_path)
                    
                    parts = filename.split('_', 1)
                    if len(parts) > 1:
                        # Escape newline characters in the script
                        script = parts[1].rsplit('.', 1)[0].replace('\n', '\\n')
                        # Append the tuple to the list
                        file_data.append((id, script, new_file_path))

        # Sort the list by the ID
        file_data.sort(key=lambda x: x[0])

        # Write the sorted data to the TSV file
        for id, script, file_path in file_data:
            tsv_file.write(f"{script}\t{file_path}\n")

    print("TSV file has been created with the scripts and file paths, sorted by ID.")

def eval():
    is_metric = IntelligibilityMetric(device='cuda')
    tsv_path = os.path.join(BASE_PATH, tsv_filename)
    i_score = is_metric.compute_intelligibility(tsv_path, None)
    

    # cv_metric = CLVPMetric(device='cuda', pretrained_path=CLVP_PATH)
    # clvp_score = cv_metric.compute_clvp(tsv_path, REF_PATH) if REF_PATH else None

    # fd_score = cv_metric.compute_fd(BASE_PATH, REF_PATH) if REF_PATH else None


    print("*Intelligibility Score:", i_score)
    # print("CLVP Score", clvp_score)
    # print("*CLVP Frechet Distance", fd_score)

    return (i_score) # , clvp_score, fd_score)

def get_step_from_name(step_name):
    return int(step_name.replace("step-", "").replace(".bin", ""))
def eval_run(ckpt_folder):
    step_names = os.listdir(ckpt_folder)
    step_names = [s for s in step_names if s.endswith(".bin")]
    step_id = [get_step_from_name(s) for s in step_names]
    results = []
    for name in step_names:
        step_ckpt_path = os.path.join(ckpt_folder, name)
        pass

if __name__ == "__main__":
    eval()
