import numpy as np
import matplotlib.pyplot as plt

# Constants for HOPG lattice
a = 0.246  # lattice constant for graphene in nm
k = 2 * np.pi / a  # wave vector
n = 1000  # size of the simulation grid
real_space_size = 1 # nm
theta = np.pi / 3  

# Function to generate harmonic wave vectors
def generate_wave_vectors(k, theta, n_harmonics):
    wave_vectors = []
    for n in range(1, n_harmonics + 1):
        kx1, ky1 = n * k, 0
        kx2, ky2 = n * k * np.cos(theta), n * k * np.sin(theta)
        wave_vectors.append((kx1, ky1, kx2, ky2))
    return wave_vectors

# Generate wave vectors for the first harmonics
n_harmonics = 5
wave_vectors = generate_wave_vectors(k, theta, n_harmonics)
P_n = [1, 0.1, 0.1, 0.1, 0.1]

# Generate real space lattice points
x = np.linspace(-real_space_size/2, real_space_size/2, n)
y = np.linspace(-real_space_size/2, real_space_size/2, n)
X, Y = np.meshgrid(x, y)
lattice_real_space = np.zeros_like(X)

# Adding contributions from A and B sites
for i, (kx1, ky1, kx2, ky2) in enumerate(wave_vectors):
    P = P_n[i]
    # A-site contribution
    lattice_real_space += P * (np.cos(kx1 * X + ky1 * Y) + np.cos(kx2 * X + ky2 * Y))
    # B-site contribution with a phase shift (e.g., pi)
    lattice_real_space += P * (np.cos(kx1 * X + ky1 * Y + np.pi/3) + np.cos(kx2 * X + ky2 * Y + np.pi/3))

sigma = 10000  # standard deviation for Gaussian smoothing
gaussian_envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
lattice_real_space_smooth = lattice_real_space

# Plot real space lattice
plt.title("HOPG Lattice in Real Space (A and B Sites)")
plt.imshow(lattice_real_space_smooth, cmap='plasma')
plt.colorbar()
plt.xticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(-real_space_size/2, real_space_size/2, 5), 2))
plt.yticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(-real_space_size/2, real_space_size/2, 5), 2))
plt.xlabel("X (nm)")
plt.ylabel("Y (nm)")
plt.savefig("Produced_Plots/FFTSIM/HOPG_Lattice_Real_Space.png", dpi=300)
plt.show()

# Plot 3D 'sideways' plot of the real space lattice
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, lattice_real_space_smooth, cmap='plasma')

ax.set_title("3D Sideways Plot of HOPG Lattice in Real Space")
ax.set_xlabel("X (nm)")
ax.set_ylabel("Y (nm)")
ax.set_zlabel("Amplitude")

plt.savefig("Produced_Plots/FFTSIM/HOPG_Lattice_Real_Space_3D.png", dpi=300)
plt.show()
# Compute 2D FFT for reciprocal space
lattice_reciprocal_space = np.fft.fftshift(np.fft.fft2(lattice_real_space_smooth))
magnitude_reciprocal = np.abs(lattice_reciprocal_space)

# Plot reciprocal space (2D FFT result)
plt.title("Reciprocal Space (2D FFT of HOPG Lattice)")
plt.imshow(np.log(1 + magnitude_reciprocal), cmap='plasma')  # 1 + data for contrast
kx = np.fft.fftshift(np.fft.fftfreq(n, d=(x[1] - x[0])))
ky = np.fft.fftshift(np.fft.fftfreq(n, d=(y[1] - y[0])))
plt.xticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(kx[0], kx[-1], 5), 2))
plt.yticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(ky[0], ky[-1], 5), 2))
plt.xlabel("kx (1/nm)")
plt.ylabel("ky (1/nm)")
plt.colorbar()
plt.savefig("Produced_Plots/FFTSIM/HOPG_Lattice_Reciprocal_Space.png", dpi=300)
plt.show()

