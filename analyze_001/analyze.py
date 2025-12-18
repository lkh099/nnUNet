import nibabel as nib
import numpy as np
import torch
import statistics as st
import os 

device = "cuda"

img_sparsity_seg_list = [[], [], [], []]
sparse_row_ratio_list = [[], [], [], []]

def count_segments_1d_cuda(row, min_val):
    """
    row: (W,) tensor on CUDA
    Returns:
      num_segments: int
      is_sparse_row: bool
    """
    # sparse = (abs(row - min_val) <= 0.001)
    sparse = (row == min_val)

    # transitions: sparse[i] != sparse[i-1]
    transitions = sparse[1:] != sparse[:-1]

    # number of segments = transitions + 1
    num_segments = transitions.sum().item() + 1

    is_sparse = sparse.all().item()  # True if whole row is sparse

    return num_segments, is_sparse


for img_idx in range(485):

    base = f"/DATA/lkh099/nnunet/nnUNet_raw/Dataset001_BrainTumour/imagesTr/BRATS_{img_idx:03d}"
    if not os.path.exists(base + "_0000.nii.gz"):
        print("no input file")
        continue

    print("Processing", img_idx)

    for i in range(4):
        img = nib.load(f"{base}_000{i}.nii.gz")
        data = img.get_fdata(dtype=np.float32)
        data = torch.from_numpy(data).to(device)   # (D, H, W) on CUDA

        is_integer_like = torch.allclose(data, data.round())
        print("Integer-like:", is_integer_like)

        min_data = data.min()   # CUDA scalar
        max_data = data.max()
        print("min = ", min_data, "max = ", max_data)
        D, H, W = data.shape

        total_segments = 0
        sparse_rows = 0

        # PROCESS EACH (D,H) LINE ON GPU
        # still loop over D,H (small loops!), but inner W loop goes away
        for d in range(D):
            for h in range(H):

                row = data[d, h, :]   # (W,) CUDA tensor

                seg_cnt, is_sparse = count_segments_1d_cuda(row, min_data)

                total_segments += seg_cnt
                if is_sparse:
                    sparse_rows += 1

        avg_segments = total_segments / (D * H)
        sparse_ratio = sparse_rows / (D * H)

        img_sparsity_seg_list[i].append(avg_segments)
        sparse_row_ratio_list[i].append(sparse_ratio)

        print("  seg=", avg_segments, " sparse_ratio=", sparse_ratio)

print("--------------------------")
print("Average per channel:")
for i in range(4):
    print("Ch", i, "segments:", st.mean(img_sparsity_seg_list[i]))
    print("Ch", i, "sparse rows:", st.mean(sparse_row_ratio_list[i]))
