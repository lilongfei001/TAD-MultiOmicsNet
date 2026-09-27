import pandas as pd
import numpy as np
import os

cell = "GM12878"
resolution = 10000
display_reso = int(resolution / 1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

def process_file(file_path):
    if not os.path.exists(file_path):
        return np.array([], dtype=np.int64)
    df = pd.read_csv(file_path, header=None, delimiter='\t')
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df = df.dropna(subset=[0])
    df[0] = df[0].astype(np.int64)
    df_np = np.array(df, dtype=np.int64).flatten()
    df_np = np.unique(df_np)
    return np.sort(df_np)


if __name__ == "__main__":
    callers = ['Arrowhead', 'DI', 'deDoc', 'SpectralTAD', 'TopDom']
    chrs = list(range(20, 23))

    output_dir = f'{root_dir}/{cell}/alpha/'
    os.makedirs(output_dir, exist_ok=True)

    all_strong_lines = []
    all_weak_lines = []

    for chr_id in chrs:
        result_dict = {}
        for caller in callers:
            file_path = f'{root_dir}/{cell}/{display_reso}kb/all_TADs/{caller}/{cell}_{display_reso}kb_{caller}.chr{chr_id}'
            result_dict[caller] = process_file(file_path)

        all_data = np.concatenate(list(result_dict.values()))
        unique_data, counts = np.unique(all_data, return_counts=True)

        strong_coords = unique_data[counts >= 3]
        weak_coords = unique_data[counts == 2]

        chr_label = f'chr{chr_id}'
        all_strong_lines.extend([f'{chr_label}\t{v}\t{v + resolution}' for v in strong_coords])
        all_weak_lines.extend([f'{chr_label}\t{v}\t{v + resolution}' for v in weak_coords])

        print(f"chr{chr_id} | Strong Boundary: {len(strong_coords)} | Weak Boundary: {len(weak_coords)}")

    print(f"\nTotal test set | Strong Boundary: {len(all_strong_lines)} | Weak Boundary: {len(all_weak_lines)}")

    with open(f'{output_dir}{cell}_{display_reso}kb_strong_boundary.txt', 'w') as f:
        f.write('\n'.join(all_strong_lines))
    with open(f'{output_dir}{cell}_{display_reso}kb_weak_boundary.txt', 'w') as f:
        f.write('\n'.join(all_weak_lines))
