import numpy as np
import matplotlib.pyplot as plt

# Constants for HOPG lattice
a = 0.246  # lattice constant for graphene in nm
k = 2 * np.pi / a  # wave vector
n = 1000  # size of the simulation grid
real_space_size =2 #nm
theta = np.pi / 3  
kx1, ky1 = k, 0
kx2, ky2 = k * np.cos(theta), k * np.sin(theta)
# Second harmonic wave vectors
kx1_2, ky1_2 = 2 * kx1, 2 * ky1
kx2_2, ky2_2 = 2 * kx2, 2 * ky2

# Third harmonic wave vectors
kx1_3, ky1_3 = 3 * kx1, 3 * ky1
kx2_3, ky2_3 = 3 * kx2, 3 * ky2
# Generate real space lattice points
x = np.linspace(-real_space_size/2, real_space_size/2, n)
y = np.linspace(-real_space_size/2, real_space_size/2, n)
X, Y = np.meshgrid(x, y)
P_1,P_2 = 1, 0.5
lattice_real_space = np.cos(kx1 * X + ky1 * Y) + np.cos(kx2 * X + ky2 * Y) +0.5 * (np.cos(kx1_2 * X + ky1_2 * Y) + np.cos(kx2_2 * X + ky2_2 * Y)) +  0.25 * (np.cos(kx1_3 * X + ky1_3 * Y) + np.cos(kx2_3 * X + ky2_3 * Y)) 
sigma = 50  # standard deviation for Gaussian smoothing
gaussian_envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
lattice_real_space_smooth = lattice_real_space * gaussian_envelope
print(lattice_real_space_smooth)
# Plot real space lattice
plt.title("HOPG Lattice in Real Space")
plt.imshow(lattice_real_space, cmap='plasma')
plt.colorbar()
plt.savefig("Produced_Plots/FFTSIM/HOPG_Lattice_Real_Space.png",dpi=300)
plt.show()


# Compute 2D FFT for reciprocal space
lattice_reciprocal_space = np.fft.fftshift(np.fft.fft2(lattice_real_space))
magnitude_reciprocal = np.abs(lattice_reciprocal_space)

# Plot reciprocal space (2D FFT result)
plt.title("Reciprocal Space (2D FFT of HOPG Lattice)")
plt.imshow(np.log(magnitude_reciprocal), cmap='plasma')
plt.colorbar()
plt.savefig("Produced_Plots/FFTSIM/HOPG_Lattice_Reciprocal_Space.png", dpi=300)
plt.show()
