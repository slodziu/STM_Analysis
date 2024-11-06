import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Function to generate harmonic wave vectors
def generate_wave_vectors(k, theta, n_harmonics):
    wave_vectors = []
    for n in range(1, n_harmonics + 1):
        kx1, ky1 = n * k, 0
        kx2, ky2 = n * k * np.cos(theta), n * k * np.sin(theta)
        kx3, ky3 = n * k * np.cos(-theta), n * k * np.sin(-theta)
        wave_vectors.append((kx1, ky1, kx2, ky2,kx3,ky3))
    return wave_vectors

def generate_coefficients(num_coefficients,order=2):
    P_n=[0.5]
    for i in range(num_coefficients):
        P_n.append(order*10**(-i))
    return P_n

def simulate_hopg_lattice(n_harmonics, k, theta, real_space_size,order,n):
    wave_vectors = generate_wave_vectors(k, theta, n_harmonics)
    P_n = generate_coefficients(n_harmonics,order)

    # Generate real space lattice points
    x = np.linspace(-real_space_size/2, real_space_size/2, n)
    y = np.linspace(-real_space_size/2, real_space_size/2, n)
    X, Y = np.meshgrid(x, y)
    lattice_real_space = np.zeros_like(X)

    # Adding contributions from A and B sites
    for i, (kx1, ky1, kx2, ky2, kx3, ky3) in enumerate(wave_vectors):
        P = P_n[i]
        # A-site contribution
        lattice_real_space += P * (np.cos((kx1 * X + ky1 * Y)) + np.cos((kx2 * X + ky2 * Y))+np.cos((kx3 * X + ky3 * Y)))

        # B-site contribution with a phase shift
        lattice_real_space += P * (np.cos((kx1 * X + ky1 * Y) + np.pi) + np.cos((kx2 * X + ky2 * Y) + np.pi)+np.cos((kx3 * X + ky3 * Y) + np.pi))

    sigma = 100  # standard deviation for Gaussian smoothing
    gaussian_envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    lattice_real_space_smooth = lattice_real_space 

    # Plot real space lattice
    plt.title("HOPG Lattice in Real Space (A and B Sites)")
    plt.imshow(lattice_real_space_smooth, cmap='plasma')
    plt.xticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(-real_space_size/2, real_space_size/2, 5), 2))
    plt.yticks(ticks=np.linspace(0, n, 5), labels=np.round(np.linspace(-real_space_size/2, real_space_size/2, 5), 2))
    plt.colorbar(label=r'Charge Density $\rho$ (AU)')
    plt.xlabel("X (nm)")
    plt.ylabel("Y (nm)")
    plt.savefig(f"Produced_Plots/FFTSIM/HOPG_Lattice_Real_Space_{real_space_size}nm.png", dpi=300)
    plt.show()

    # Plot 3D 'sideways' plot of the real space lattice
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, lattice_real_space_smooth, cmap='plasma')
    fig.colorbar(surface, label=r'Charge Density $\rho$ (AU)')
    ax.set_title("3D Sideways Plot of HOPG Lattice in Real Space")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Amplitude")

    plt.savefig(f"Produced_Plots/FFTSIM/HOPG_Lattice_Real_Space_3D_{real_space_size}nm.png", dpi=300)
    plt.show()

    # Compute 2D FFT for reciprocal space
    lattice_reciprocal_space = np.fft.fftshift(np.fft.fft2(lattice_real_space_smooth))
    magnitude_reciprocal = np.abs(lattice_reciprocal_space)
    # Plot reciprocal space (2D FFT result)
    plt.title("k-space of Simulated HOPG Lattice (2D FFT)")
    plt.imshow(np.log(1 + magnitude_reciprocal), cmap='plasma')  # 1 + data for contrast
    kx = np.fft.fftshift(np.fft.fftfreq(n, d=(x[1] - x[0])))
    ky = np.fft.fftshift(np.fft.fftfreq(n, d=(y[1] - y[0])))

    plt.xlabel(r"$k_x$ (1/nm)")
    plt.ylabel(r"$k_y$ (1/nm)")
    scale_length = 60
    plt.xlim(n//2-scale_length,n//2+scale_length)
    plt.ylim(n//2-scale_length,n//2+scale_length)
    plt.xticks(ticks=np.linspace(n//2-scale_length, n//2+scale_length, 5), labels=np.round(np.linspace(kx[n//2-scale_length], kx[n//2+scale_length], 5), 1))
    plt.yticks(ticks=np.linspace(n//2-scale_length, n//2+scale_length, 5), labels=np.round(np.linspace(ky[n//2-scale_length], ky[n//2+scale_length], 5), 1))
    plt.colorbar(label=r'\log{Intensity} (AU)')
    plt.savefig(f"Produced_Plots/FFTSIM/HOPG_Lattice_Reciprocal_Space_{real_space_size}nm.png", dpi=300)
    plt.show()


# Constants for HOPG lattice
a = 0.246  # lattice constant for graphene in nm
k = 2 * np.pi / a  # wave vector
n_pts = 3000 # size of the simulation grid
real_space_size = 3 # nm
theta = np.pi / 3  
n_harmonics = 10  # number of harmonics to include in the lattice
simulate_hopg_lattice(n_harmonics, k, theta, real_space_size,10,n_pts)
