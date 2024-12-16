import numpy as np

# Load the Udofs.txt file, which contains values in scientific notation
# Set dtype to float initially to correctly handle scientific notation
Udofs = np.loadtxt(r"M:\constantinos\data\bio\5-equations\dofs\Udofs.txt", dtype=float)

# Print the raw loaded values for inspection
print("Raw Udofs values (before rounding):", Udofs)

# Convert to integers after rounding to handle any small floating-point errors
Udofs = np.round(Udofs).astype(int)

# Print the rounded and converted values
print("Udofs values (after rounding):", Udofs)

# Check for out-of-bound values or duplicates, or any other processing as before
