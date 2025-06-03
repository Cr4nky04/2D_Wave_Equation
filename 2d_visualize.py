import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Get sorted list of CSV files
file_list = sorted(glob.glob("wave_step_*.csv"))

# Load one frame to get shape
data_sample = np.loadtxt(file_list[0], delimiter=',')
nx, ny = data_sample.shape
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
X, Y = np.meshgrid(x, y)

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial surface plot
surf = [ax.plot_surface(X, Y, data_sample, cmap='viridis', edgecolor='k')]

def update(frame):
    global surf
    ax.clear()
    
    data = np.loadtxt(file_list[frame], delimiter=',')

    # Update surface plot
    surf[0] = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='k', rstride=1, cstride=1, linewidth=0)
    ax.set_zlim(-1, 1)
    ax.set_title(f"2D Wave at step {file_list[frame].split('_')[-1].split('.')[0]}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Amplitude')

    return surf

ani = animation.FuncAnimation(fig, update, frames=len(file_list), interval=100, blit=False)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Get sorted list of CSV files
file_list = sorted(glob.glob("wave_step_*.csv"))

# Load one frame to get shape
data_sample = np.loadtxt(file_list[0], delimiter=',')
nx, ny = data_sample.shape
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
X, Y = np.meshgrid(x, y)

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial surface plot
surf = [ax.plot_surface(X, Y, data_sample, cmap='viridis', edgecolor='k')]

def update(frame):
    global surf
    ax.clear()
    
    data = np.loadtxt(file_list[frame], delimiter=',')

    # Update surface plot
    surf[0] = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='k', rstride=1, cstride=1, linewidth=0)
    ax.set_zlim(-1, 1)
    ax.set_title(f"2D Wave at step {file_list[frame].split('_')[-1].split('.')[0]}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Amplitude')

    return surf

ani = animation.FuncAnimation(fig, update, frames=len(file_list), interval=100, blit=False)

plt.show()
