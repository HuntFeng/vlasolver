# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

# %%
x = np.linspace(-1,1,9)
dx = x[1] - x[0]
eta = x**2 - 0.7**2
# eta = -x**2 + 0.7**2
print(x)
print(eta)


# %%
def compute_theta1(i: int):
    dx_eta = (eta[i+1] - eta[i-1]) / 2.0
    dxx_eta = (eta[i+1] - 2*eta[i] + eta[i-1]) / 2.0
    return  (-dx_eta -np.sqrt(dx_eta**2 - 4*dxx_eta*eta[i])) / (2*dxx_eta)
    
def compute_theta2(i: int):
    dx_eta = (eta[i+1] - eta[i-1]) / 2.0
    dxx_eta = (eta[i+1] - 2*eta[i] + eta[i-1]) / 2.0
    return  (-dx_eta +np.sqrt(dx_eta**2 - 4*dxx_eta*eta[i])) / (2*dxx_eta)

def compute_theta(i: int, direction: str):
    dx_eta = (eta[i+1] - eta[i-1]) / 2.0
    dxx_eta = (eta[i+1] - 2*eta[i] + eta[i-1]) / 2.0
    if direction == "L":
        return  (dx_eta - np.sign(eta[i])* np.sqrt(dx_eta**2 - 4*dxx_eta*eta[i])) / (2*dxx_eta)
    elif direction == "R":
        return  (-dx_eta - np.sign(eta[i])* np.sqrt(dx_eta**2 - 4*dxx_eta*eta[i])) / (2*dxx_eta)
    else:
        raise Exception("no such direction")
    


# %%
print(x[1] + compute_theta(1, direction="R") * dx)
print(x[2] - compute_theta(2, direction="L") * dx )
print(x[6] + compute_theta(6, direction="R") * dx)
print(x[7] - compute_theta(7, direction="L") * dx)

# %%
print(f"R, eta={eta[1]}, f1={compute_theta1(1) * dx}")
print(f"L, eta={eta[2]}, f1={compute_theta1(2) * dx}")
print(f"R, eta={eta[1]}, f2={compute_theta2(1) * dx}")
print(f"L, eta={eta[2]}, f2={compute_theta2(2) * dx}")

# %%
print(f"R, eta={eta[6]}, f1={compute_theta1(6) * dx}")
print(f"L, eta={eta[7]}, f1={compute_theta1(7) * dx}")
print(f"R, eta={eta[6]}, f2={compute_theta2(6) * dx}")
print(f"L, eta={eta[7]}, f2={compute_theta2(7) * dx}")

# %%
