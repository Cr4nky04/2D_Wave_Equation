#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 100
#define NY 100
#define STEPS 100000
#define SAVE_INTERVAL 10

const double c = 1.0;
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

// Allocate host 2D array for saving
double** allocate_2d_array_host(int nx, int ny) {
    double** arr = (double**)cudamalloc(nx * sizeof(double*));
    if (!arr) { fprintf(stderr, "Allocation failed\n"); exit(1); }
    for (int i = 0; i < nx; i++) {
        arr[i] = (double*)cudamalloc(ny * sizeof(double));
        if (!arr[i]) { fprintf(stderr, "Allocation failed\n"); exit(1); }
    }
    return arr;
}

void free_2d_array_host(double** arr, int nx) {
    for (int i = 0; i < nx; i++) free(arr[i]);
    free(arr);
}

void save_to_csv(double** arr, int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "cuda_wave_step_%04d.csv", step);
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

// CUDA kernels ----------------------------------

// Initialize wave: u_prev = 0, u_next = 0, u_curr = Gaussian
__global__ void initialize_wave_kernel(double* u_prev, double* u_curr, double* u_next,
                                       int nx, int ny, double dx, double dy, double sigma) {
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
        u_curr[idx] = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
    }
}

// Initialize u_prev for first step (zero initial velocity)
__global__ void initialize_prev_kernel(double* u_prev, double* u_curr, int nx, int ny, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i * ny + j;
        int idx_right = (i + 1) * ny + j;
        int idx_left = (i - 1) * ny + j;
        int idx_up = i * ny + (j + 1);
        int idx_down = i * ny + (j - 1);

        u_prev[idx] = u_curr[idx] - 0.5 * r * r * (
            u_curr[idx_right] + u_curr[idx_left] + u_curr[idx_up] + u_curr[idx_down] - 4 * u_curr[idx]
        );
    }
}

// Wave update kernel for each step
__global__ void wave_step_kernel(double* u_prev, double* u_curr, double* u_next,
                                 int nx, int ny, double r) {
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

// Apply zero boundary conditions
__global__ void apply_boundary_conditions_kernel(double* u, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx) {
        u[idx * ny + 0] = 0.0;
        u[idx * ny + ny - 1] = 0.0;
    }
    if (idx < ny) {
        u[0 * ny + idx] = 0.0;
        u[(nx - 1) * ny + idx] = 0.0;
    }
}

// ---------------------------------------------

int main() {
    r = c * dt / dx;
    if (r >= 1 / sqrt(2)) {
        printf("Warning: Courant condition not met (r=%f). Reduce dt or increase dx.\n", r);
    }

    size_t size = NX * NY * sizeof(double);

    // Allocate device arrays
    double *d_u_prev, *d_u_curr, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, size));
    CUDA_CHECK(cudaMalloc(&d_u_curr, size));
    CUDA_CHECK(cudaMalloc(&d_u_next, size));

    dim3 blockSize(16, 16);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    // Initialize on device
    initialize_wave_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, NX, NY, dx, dy, 0.05);
    CUDA_CHECK(cudaDeviceSynchronize());

    initialize_prev_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, NX, NY, r);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host 2D array for saving CSV
    double** h_u_curr = allocate_2d_array_host(NX, NY);

    for (int step = 1; step <= STEPS; step++) {
        wave_step_kernel<<<gridSize, blockSize>>>(d_u_prev, d_u_curr, d_u_next, NX, NY, r);
        CUDA_CHECK(cudaDeviceSynchronize());

        int maxDim = NX > NY ? NX : NY;
        int block1D = 256;
        int grid1D = (maxDim + block1D - 1) / block1D;

        apply_boundary_conditions_kernel<<<grid1D, block1D>>>(d_u_next, NX, NY);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        double* temp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = temp;

        // Save every SAVE_INTERVAL steps
        if (step % SAVE_INTERVAL == 0) {
            double* flat_arr = (double*)malloc(size);
            CUDA_CHECK(cudaMemcpy(flat_arr, d_u_curr, size, cudaMemcpyDeviceToHost));

            for (int i = 0; i < NX; i++)
                for (int j = 0; j < NY; j++)
                    h_u_curr[i][j] = flat_arr[i * NY + j];

            free(flat_arr);
            save_to_csv(h_u_curr, step);
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
