#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 1000
#define NY 1000
#define STEPS 10000
#define SAVE_INTERVAL 1000  // Save every 100 steps

const double c = 1.0;  // Speed of the wave
const double dx = 0.01;
const double dy = 0.01;
const double dt = 0.005;

double r;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

double** allocate_2d_array_host(int nx, int ny) {
    double** arr = (double**)malloc(nx * sizeof(double*));
    if (!arr) { fprintf(stderr, "Allocation failed\n"); exit(1); }
    for (int i = 0; i < nx; i++) {
        arr[i] = (double*)malloc(ny * sizeof(double));
        if (!arr[i]) { fprintf(stderr, "Allocation failed\n"); exit(1); }
    }
    return arr;
}

void free_2d_array_host(double** arr, int nx) {
    for (int i = 0; i < nx; i++) free(arr[i]);
    free(arr);
}

// CUDA Kernel: Initialize wave matrix (same as in acoustic wave equation)
__global__ void initialize_wave_kernel(double* u_prev, double* u_curr, double* u_next, int nx, int ny, double dx, double dy, double sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = nx / 2;
    int cy = ny / 2;

    if (i < nx && j < ny) {
        int idx = i * ny + j;
        double x = (i - cx) * dx;
        double y = (j - cy) * dy;
        u_prev[idx] = 0.0;
        u_next[idx] = 0.0;
        u_curr[idx] = exp(-(x * x + y * y) / (2.0 * sigma * sigma)); // Gaussian wave
    }
}

// CUDA Kernel: Update wave matrix in x-direction (half-step)
__global__ void wave_step_x_kernel(double* u_prev, double* u_curr, double* u_next, int nx, int ny, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i * ny + j;
        int idx_right = (i + 1) * ny + j;
        int idx_left = (i - 1) * ny + j;
        int idx_up = i * ny + (j + 1);
        int idx_down = i * ny + (j - 1);

        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + r * r * (
            u_curr[idx_right] + u_curr[idx_left] + u_curr[idx_up] + u_curr[idx_down] - 4.0 * u_curr[idx]
        );
    }
}

// CUDA Kernel: Update wave matrix in y-direction (half-step)
__global__ void wave_step_y_kernel(double* u_prev, double* u_curr, double* u_next, int nx, int ny, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i * ny + j;
        int idx_right = (i + 1) * ny + j;
        int idx_left = (i - 1) * ny + j;
        int idx_up = i * ny + (j + 1);
        int idx_down = i * ny + (j - 1);

        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + r * r * (
            u_curr[idx_right] + u_curr[idx_left] + u_curr[idx_up] + u_curr[idx_down] - 4.0 * u_curr[idx]
        );
    }
}

// CUDA Kernel: Apply boundary conditions
__global__ void apply_boundary_conditions_kernel(double* u, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx) {
        u[idx * ny + 0] = 0.0;    // Zero boundary at the top
        u[idx * ny + ny - 1] = 0.0; // Zero boundary at the bottom
    }
    if (idx < ny) {
        u[0 * ny + idx] = 0.0;     // Zero boundary at the left
        u[(nx - 1) * ny + idx] = 0.0; // Zero boundary at the right
    }
}

// Save the wave matrix to CSV file (host function)
void save_to_csv(double** arr, int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "cuda_adi_wave_step_%04d.csv", step);
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s\n", filename);
        return;
    }
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(f, "%.6f", arr[i][j]);
            if (j < NY - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main() {
    r = c * dt / dx;  // Courant number
    if (r >= 1 / sqrt(2)) {
        printf("Warning: Courant condition not met (r=%f). Reduce dt or increase dx.\n", r);
    }

    size_t size = NX * NY * sizeof(double);

    // Allocate device memory (GPU)
    double *d_u_prev, *d_u_curr, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, size));
    CUDA_CHECK(cudaMalloc(&d_u_curr, size));
    CUDA_CHECK(cudaMalloc(&d_u_next, size));

    dim3 blockSize(256, 256);  // 16x16 threads per block
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    // Initialize the wave on the device
    initialize_wave_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, NX, NY, dx, dy, 0.05);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host array for saving the results (CSV)
    double** h_u_curr = allocate_2d_array_host(NX, NY);

    for (int step = 1; step <= STEPS; step++) {
        // Update wave in the x-direction (half step)
        wave_step_x_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, NX, NY, r);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply boundary conditions in the x-direction
        apply_boundary_conditions_kernel<<<gridSize, blockSize>>>(d_u_next, NX, NY);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update wave in the y-direction (half step)
        wave_step_y_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, NX, NY, r);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers for the next iteration
        double* temp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = temp;

        // Save output every SAVE_INTERVAL steps
        if (step % SAVE_INTERVAL == 0) {
            // Copy data from device to host
            double* flat_arr = (double*)malloc(size);
            CUDA_CHECK(cudaMemcpy(flat_arr, d_u_curr, size, cudaMemcpyDeviceToHost));

            // Save to CSV
            for (int i = 0; i < NX; i++) 
                for (int j = 0; j < NY; j++) 
                    h_u_curr[i][j] = flat_arr[i * NY + j];

            free(flat_arr);  // Free host memory
            save_to_csv(h_u_curr, step);  // Save to CSV
        }
    }

    // Free memory
    free_2d_array_host(h_u_curr, NX);
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u_curr));
    CUDA_CHECK(cudaFree(d_u_next));

    printf("Simulation completed.\n");
    return 0;
}
