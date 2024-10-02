import numpy as np
import sympy as sp

# Define the matrix A and vector b
A = np.array([[4, 8, -5], [-3, -6, -7], [2, 4, 2]])
b = np.array([-1, -1, 3])

# (a) Compute the reduced echelon form of A
# Use sympy to compute the reduced row echelon form (RREF)
A_sympy = sp.Matrix(A)
rref_A, pivot_columns = A_sympy.rref()  # rref returns the RREF and the pivot columns
rref_A_np = np.array(rref_A)  # Convert back to numpy array

print("Reduced Row Echelon Form of A:")
print(rref_A_np)

# (b) Find the column space of A
# The column space is spanned by the columns of A that correspond to the pivot columns in RREF
column_space = A[:, pivot_columns]

print("\nColumn Space of A:")
print(column_space)

# (c) Solve the matrix equation Ax = b
# Use numpy's linear algebra solver to find x
x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("\nLeast Squares Solution to the equation Ax = b:")
print(x_ls)

# (d) Compute the Null Space of A
# We use sympy to compute the null space
null_space = A_sympy.nullspace()
null_space_np = [np.array(v).astype(float) for v in null_space]  # Convert each vector to numpy array

print("\nNull Space of A:")
for vector in null_space_np:
    print(vector)
