import numpy as np

# Load Udofs.txt and Pdofs.txt
Udofs = np.loadtxt(r"M:\constantinos\data\bio\5-equations\dofs\Udofs.txt").astype(int)
Pdofs = np.loadtxt(r"M:\constantinos\data\bio\5-equations\dofs\Pdofs.txt").astype(int)

# Combine the two vectors
combined_dofs = np.concatenate((Udofs, Pdofs))

# Check if the indices are zero-based
is_zero_based = combined_dofs.min() == 0
print(f"Is the combined DOF set zero-based? {is_zero_based}")

# Check if the combined set has 2368 consecutive values
expected_range = np.arange(2368)  # This is the range from 0 to 2367 (2368 total)
are_consecutive = np.array_equal(np.sort(combined_dofs), expected_range)
print(
    f"Does the combined DOF set contain all consecutive values from 0 to 2367? {are_consecutive}"
)

# Check for duplicates
are_duplicates = len(combined_dofs) != len(np.unique(combined_dofs))
print(f"Are there duplicates in the combined DOF set? {are_duplicates}")

# Output additional information
if not is_zero_based:
    print(f"The minimum value in the combined DOF set is: {combined_dofs.min()}")
if not are_consecutive:
    print(
        f"The missing or extra values are: {set(expected_range) ^ set(combined_dofs)}"
    )  # Symmetric difference
if are_duplicates:
    print(
        f"The duplicate values are: {combined_dofs[np.where(np.bincount(combined_dofs) > 1)]}"
    )

a = 1
