#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 100
#define NY 100
#define STEPS 100000
#define SAVE_INTERVAL 10   // Save every 100 steps

const double c = 1.0;
const double dx = 0.01;
const double dy = 0.01;
const double dt = 0.005;

double r;

double** u_prev;
double** u_curr;
double** u_next;

double** allocate_2d_array(int nx, int ny) {
    double** arr = cudamalloc(nx * sizeof(double*));
    if (!arr) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < nx; i++) {
        arr[i] = cudamalloc(ny * sizeof(double));
        if (!arr[i]) {
            fprintf(stderr, "Allocation failed\n");
            exit(1);
        }
    }
    return arr;
}

void free_2d_array(double** arr, int nx) {
    for (int i = 0; i < nx; i++) {
        free(arr[i]);
    }
    free(arr);
}

void initialize_wave() {
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.05;

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x = (i - cx) * dx;
            double y = (j - cy) * dy;
            u_prev[i][j] = 0.0;
            u_next[i][j] = 0.0;
            u_curr[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
        }
    }

    // First time step (zero initial velocity)
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            u_prev[i][j] = u_curr[i][j] - 0.5 * r * r * (
                u_curr[i + 1][j] + u_curr[i - 1][j] + u_curr[i][j + 1] + u_curr[i][j - 1] - 4 * u_curr[i][j]
                );
        }
    }
}

// Save current wave state to CSV file
void save_to_csv(double** arr, int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "c_wave_step_%04d.csv", step);

    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
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
    r = c * dt / dx;
    if (r >= 1 / sqrt(2)) {
        printf("Warning: Courant condition not met (r=%f). Reduce dt or increase dx.\n", r);
    }

    u_prev = allocate_2d_array(NX, NY);
    u_curr = allocate_2d_array(NX, NY);
    u_next = allocate_2d_array(NX, NY);

    initialize_wave();

    for (int step = 1; step <= STEPS; step++) {
        // Update wave
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                u_next[i][j] = 2 * u_curr[i][j] - u_prev[i][j] + r * r * (
                    u_curr[i + 1][j] + u_curr[i - 1][j] + u_curr[i][j + 1] + u_curr[i][j - 1] - 4 * u_curr[i][j]
                    );
            }
        }

        // Boundary conditions: fixed zero
        for (int i = 0; i < NX; i++) {
            u_next[i][0] = 0.0;
            u_next[i][NY - 1] = 0.0;
        }
        for (int j = 0; j < NY; j++) {
            u_next[0][j] = 0.0;
            u_next[NX - 1][j] = 0.0;
        }

        // Swap pointers
        double** temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;

        // Save output every SAVE_INTERVAL steps
        if (step % SAVE_INTERVAL == 0) {
            save_to_csv(u_curr, step);
        }
    }

    free_2d_array(u_prev, NX);
    free_2d_array(u_curr, NX);
    free_2d_array(u_next, NX);

    printf("Simulation completed.\n");
    return 0;
}
