import numpy as np
import pandas as pd


def calculate_ep_l(df):
    ep_l = []
    for i in range(len(df)):
        if (
            df["d_{i+1/2}"].get(i - 1, np.nan) < 0
            and df["d_{i+1/2}"].get(i, np.nan) <= 0
        ):
            ep_l.append(
                np.min(
                    [
                        1.0,
                        df["delta_i"].get(i, np.nan)
                        / df["d_{i+1/2}"].get(i - 1, np.nan),
                    ],
                )
            )
        elif (
            df["d_{i+1/2}"].get(i - 1, np.nan) * df["d_{i+1/2}"].get(i, np.nan) < 0
            and df["p_i"].get(i, np.nan) < 0
        ):
            ep_l.append(
                df["delta_i"].get(i, np.nan)
                / (df["d_{i+1/2}"].get(i - 1, np.nan) - df["d_{i+1/2}"].get(i, np.nan))
            )
        else:
            ep_l.append(1.0)
    return ep_l


def calculate_ep_r(df):
    ep_r = []
    for i in range(len(df)):
        if (
            df["d_{i+1/2}"].get(i - 1, np.nan) >= 0
            and df["d_{i+1/2}"].get(i, np.nan) > 0
        ):
            ep_r.append(
                np.min(
                    [
                        1.0,
                        -df["delta_i"].get(i, np.nan) / df["d_{i+1/2}"].get(i, np.nan),
                    ]
                )
            )
        elif (
            df["d_{i+1/2}"].get(i - 1, np.nan) * df["d_{i+1/2}"].get(i, np.nan) < 0
            and df["p_i"].get(i, np.nan) < 0
        ):
            ep_r.append(
                df["delta_i"].get(i, np.nan)
                / (df["d_{i+1/2}"].get(i - 1, np.nan) - df["d_{i+1/2}"].get(i, np.nan))
            )
        else:
            ep_r.append(1.0)
    return ep_r


def calculate_ep(df):
    ep = []
    for i in range(len(df)):
        # at surface cell, don't consider limiter from the ghost cell
        if i == 3:
            ep.append(df["ep_r"].get(i, np.nan))
        else:
            ep.append(
                np.min([df["ep_r"].get(i, np.nan), df["ep_l"].get(i + 1, np.nan)])
            )
    return ep


non_negative_extrapolation = False
nu = 0.1
df = pd.DataFrame({"f_i": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
# df = pd.DataFrame({"f_i": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
# extrapolation
if non_negative_extrapolation:
    df.loc[4, "f_i"] = np.max([2 * df.loc[3, "f_i"] - df.loc[2, "f_i"], 0.0])
else:
    df.loc[4, "f_i"] = 2 * df.loc[3, "f_i"] - df.loc[2, "f_i"]
# if non_negative_extrapolation:
#     df.loc[1, "f_i"] = np.max([2 * df.loc[2, "f_i"] - df.loc[3, "f_i"], 0.0])
# else:
#     df.loc[1, "f_i"] = 2 * df.loc[2, "f_i"] - df.loc[3, "f_i"]
df.index.name = "i"
df["f_{i+1/2}"] = df["f_i"] * nu
df["delta_i"] = [
    -df["f_i"].get(i, np.nan)
    - df["f_{i+1/2}"].get(i - 1, np.nan)
    + df["f_{i+1/2}"].get(i, np.nan)
    for i in range(len(df))
]
df["F_{i+1/2}"] = [
    nu
    * (
        df["f_i"].get(i, np.nan)
        + (1 / 6)
        * (2 - abs(nu))
        * (1 - abs(nu))
        * (df["f_i"].get(i + 1, np.nan) - df["f_i"].get(i, np.nan))
        + (1 / 6)
        * (1 - abs(nu))
        * (1 + abs(nu))
        * (df["f_i"].get(i, np.nan) - df["f_i"].get(i - 1, np.nan))
    )
    for i in range(len(df))
]

df["d_{i+1/2}"] = df["F_{i+1/2}"] - df["f_{i+1/2}"]
df["p_i"] = [
    df["d_{i+1/2}"].get(i - 1, np.nan)
    - df["d_{i+1/2}"].get(i, np.nan)
    - df["delta_i"].get(i, np.nan)
    for i in range(len(df))
]
df["ep_l"] = calculate_ep_l(df)
df["ep_r"] = calculate_ep_r(df)
df["ep_{i+1/2}"] = calculate_ep(df)
df["F^hat_{i+1/2}"] = (
    df["ep_{i+1/2}"] * (df["F_{i+1/2}"] - df["f_{i+1/2}"]) + df["f_{i+1/2}"]
)
df["f_i_new"] = [
    df["f_i"].get(i, np.nan)
    + df["F^hat_{i+1/2}"].get(i - 1, np.nan)
    - df["F^hat_{i+1/2}"].get(i, np.nan)
    for i in range(len(df))
]

print(f"Advection equation: df/dt + {nu}df/dx = 0")
print(f"Enforce non-negative extrapolation: {non_negative_extrapolation}")
print("PFC update with Immersed Boundary")
print("Fluid cells from i=0 to i=3, ghost cell at i=4, and solid cell at i=5")
print("The following table shows the negative distribution happens at i=2 and/or i=3")
print()
print(df)
