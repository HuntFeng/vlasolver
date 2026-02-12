import matplotlib.pyplot as plt
import numpy as np

# Grid parameters
nx, ny = 160, 50
Lx, Ly = 1.0, 0.5
G = 3
dx = Lx / nx
dy = Ly / ny
x = np.arange((0.5 - G) * dx, Lx + G * dx, dx)
y = np.arange((0.5 - G) * dy, Ly + G * dy, dy)

# Number of cells
nx_cells = len(x)
ny_cells = len(y)

# Create a base image - light gray background for the mesh
img = np.ones((ny_cells, nx_cells)) * np.nan

# Calculate which cells are in the cylinder
cylinder_mask = np.zeros_like(img, dtype=bool)
for j in range(ny_cells):
    for i in range(nx_cells):
        # Cell center coordinates
        cell_x = x[i]
        cell_y = y[j]
        # Check if the cell is inside the half circle: (x-0.375)^2 + y^2 <= 0.125^2
        # AND if y >= 0 (to make it a half circle)
        if (cell_x - 0.375) ** 2 + cell_y**2 < 0.125**2:
            cylinder_mask[j, i] = True

# Mark cells in the cylinder
img[cylinder_mask] = 0.5  # Darker shade for cells in the cylinder

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Create index-based coordinates for pcolormesh
x_idx = np.arange(nx_cells + 1) - 0.5
y_idx = np.arange(ny_cells + 1) - 0.5

# Use pcolormesh with visible cell edges
mesh = ax.pcolormesh(x_idx, y_idx, img, cmap="Greys", edgecolors="black", linewidth=0.1)
ax.pcolormesh(np.arange(nx_cells), np.arange(ny_cells), img, alpha=0.5)

# Add circle outline in the index space
center_x_idx = 0.375 / dx + G - 0.5
center_y_idx = 0 / dy + G - 0.5
radius_x_idx = 0.125 / dx
radius_y_idx = 0.125 / dy
theta = np.linspace(0, np.pi, 1000)
circle_x = center_x_idx + radius_x_idx * np.cos(theta)
circle_y = center_y_idx + radius_y_idx * np.sin(theta)
ax.plot(circle_x, circle_y, "r-", linewidth=2)

# Set axis labels and title
ax.set_xlabel("X Index")
ax.set_ylabel("Y Index")
ax.set_title("Mesh Grid with Half Circle (Index Space)")

# Add axis ticks for reference
ax.set_xticks(np.arange(0, nx_cells, 10))
ax.set_yticks(np.arange(0, ny_cells, 5))

# Add text to show the correspondence between indices and physical coordinates
text_info = (
    f"Physical grid: x ∈ [{x.min():.3f}, {x.max():.3f}], "
    f"y ∈ [{y.min():.3f}, {y.max():.3f}]\n"
    f"Center of cylinder at physical (0.375, 0) → index ({center_x_idx}, {center_y_idx})"
)
plt.figtext(0.5, 0.01, text_info, ha="center", fontsize=10)

# Set the aspect ratio to make cells appear square
ax.set_aspect("equal")

# Adjust axis limits to show the full grid
ax.set_xlim(-0.5, nx_cells - 0.5)
ax.set_ylim(-0.5, ny_cells - 0.5)
plt.show()
