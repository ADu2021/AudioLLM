

def rotate_tsv_entries(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Split each line into script and path and store separately
    scripts = [line.split('\t')[0] for line in lines if line.strip()]
    paths = [line.split('\t')[1].strip() for line in lines if line.strip()]

    # Rotate the paths list by one position
    paths = [paths[-1]] + paths[:-1]

    with open(output_file, 'w', encoding='utf-8') as file:
        for script, path in zip(scripts, paths):
            file.write(f"{script}\t{path}\n")

    print("New TSV file has been created with rotated paths.")
# Specify the path to your input and output files
input_path = '/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ckpt-2024-08-12_00-20-39-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-2_tgt_mask-lookbehind-2_mem_mask/answer/val_50.tsv'
output_path = '/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ckpt-2024-08-12_00-20-39-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-2_tgt_mask-lookbehind-2_mem_mask/answer/rotate_val_50.tsv'

rotate_tsv_entries(input_path, output_path)