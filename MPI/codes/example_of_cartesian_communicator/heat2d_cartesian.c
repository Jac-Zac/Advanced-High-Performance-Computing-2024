/*
 * MPI Cartesian Communicator Example: 2D Heat Diffusion
 * ======================================================
 * 
 * This code demonstrates the use of MPI Cartesian topologies
 * for a simple 2D heat diffusion (Laplacian) solver using
 * explicit finite differences.
 * 
 * Compile: mpicc -O2 -o heat2d heat2d_cartesian.c -lm
 * Run:     mpirun -np 4 ./heat2d
 * 
 * Learning objectives:
 *   1. MPI_Dims_create for balanced decomposition
 *   2. MPI_Cart_create to build topology
 *   3. MPI_Cart_coords/MPI_Cart_rank for translation
 *   4. MPI_Cart_shift for neighbor identification
 *   5. Derived datatypes for column exchange
 *   6. MPI_PROC_NULL handling at boundaries
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Global simulation parameters */
#define NX_GLOBAL 100    /* Global grid size in x */
#define NY_GLOBAL 100    /* Global grid size in y */
#define NSTEPS 1000      /* Number of time steps */
#define ALPHA 0.25       /* Diffusion coefficient (dt * D / dx^2) */

/* Macro for 2D array access with ghost zones */
#define U(i, j) u[(i) * ny + (j)]
#define U_NEW(i, j) u_new[(i) * ny + (j)]

int main(int argc, char **argv) {
    int world_rank, world_size;
    int cart_rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* ================================================================
     * STEP 1: Create the Cartesian topology
     * ================================================================ */
    
    /* Let MPI choose a balanced 2D decomposition */
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    
    if (world_rank == 0) {
        printf("=== MPI Cartesian Communicator Demo ===\n");
        printf("World size: %d processes\n", world_size);
        printf("Cartesian topology: %d x %d\n", dims[0], dims[1]);
        printf("Global grid: %d x %d\n", NX_GLOBAL, NY_GLOBAL);
        printf("Time steps: %d\n\n", NSTEPS);
    }
    
    /* Non-periodic boundaries (Dirichlet BC: u=0 at boundaries) */
    int periods[2] = {0, 0};
    
    /* reorder=1 allows MPI to optimize process placement */
    int reorder = 1;
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    /* Handle case where process is not part of the Cartesian grid */
    if (cart_comm == MPI_COMM_NULL) {
        if (world_rank == 0) {
            printf("WARNING: Some processes not in Cartesian grid!\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    /* Get my rank in the Cartesian communicator (may differ from world_rank!) */
    MPI_Comm_rank(cart_comm, &cart_rank);
    
    /* ================================================================
     * STEP 2: Get my coordinates and compute local domain
     * ================================================================ */
    
    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int px = coords[0];  /* My x-coordinate in process grid */
    int py = coords[1];  /* My y-coordinate in process grid */
    
    /* Compute local domain size (distribute evenly, handle remainder) */
    int nx_local = NX_GLOBAL / dims[0];
    int ny_local = NY_GLOBAL / dims[1];
    
    /* Give extra points to lower-indexed processes */
    if (px < (NX_GLOBAL % dims[0])) nx_local++;
    if (py < (NY_GLOBAL % dims[1])) ny_local++;
    
    /* Total local size including ghost zones (+1 on each side) */
    int nx = nx_local + 2;
    int ny = ny_local + 2;
    
    printf("Rank %d (world %d): coords=(%d,%d), local grid=%dx%d (with ghosts: %dx%d)\n",
           cart_rank, world_rank, px, py, nx_local, ny_local, nx, ny);
    
    /* ================================================================
     * STEP 3: Identify neighbors using MPI_Cart_shift
     * ================================================================ */
    
    int left, right, down, up;
    
    /* 
     * MPI_Cart_shift(comm, direction, displacement, &source, &dest)
     * 
     * For direction=0 (x-axis) and disp=+1:
     *   source = rank to receive FROM (my left neighbor)
     *   dest   = rank to send TO (my right neighbor)
     * 
     * At boundaries, MPI_PROC_NULL is returned.
     */
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    
    printf("Rank %d neighbors: left=%d, right=%d, down=%d, up=%d\n",
           cart_rank, left, right, down, up);
    
    /* ================================================================
     * STEP 4: Create derived datatype for column exchange
     * ================================================================ */
    
    /* 
     * Rows are contiguous in memory (can use MPI_DOUBLE directly).
     * Columns are strided: element (i,j) and (i+1,j) are 'ny' apart.
     * 
     * MPI_Type_vector(count, blocklength, stride, oldtype, newtype)
     *   count = number of blocks (nx_local rows)
     *   blocklength = 1 element per block
     *   stride = ny elements between consecutive blocks
     */
    MPI_Datatype column_type;
    MPI_Type_vector(nx_local,   /* count */
                    1,          /* blocklength */
                    ny,         /* stride */
                    MPI_DOUBLE,
                    &column_type);
    MPI_Type_commit(&column_type);
    
    /* ================================================================
     * STEP 5: Allocate arrays and initialize
     * ================================================================ */
    
    double *u     = (double *)calloc(nx * ny, sizeof(double));
    double *u_new = (double *)calloc(nx * ny, sizeof(double));
    
    if (!u || !u_new) {
        fprintf(stderr, "Rank %d: Memory allocation failed!\n", cart_rank);
        MPI_Abort(cart_comm, 1);
    }
    
    /* Initialize with a hot spot in the center of the global domain */
    /* Each process checks if its local domain contains the center */
    int global_cx = NX_GLOBAL / 2;
    int global_cy = NY_GLOBAL / 2;
    
    /* Compute my global offset */
    int offset_x = 0, offset_y = 0;
    for (int p = 0; p < px; p++) {
        int n = NX_GLOBAL / dims[0];
        if (p < (NX_GLOBAL % dims[0])) n++;
        offset_x += n;
    }
    for (int p = 0; p < py; p++) {
        int n = NY_GLOBAL / dims[1];
        if (p < (NY_GLOBAL % dims[1])) n++;
        offset_y += n;
    }
    
    /* Check if hot spot is in my local domain */
    int local_cx = global_cx - offset_x + 1;  /* +1 for ghost zone */
    int local_cy = global_cy - offset_y + 1;
    
    if (local_cx >= 1 && local_cx <= nx_local &&
        local_cy >= 1 && local_cy <= ny_local) {
        /* Set initial temperature at center */
        U(local_cx, local_cy) = 100.0;
        printf("Rank %d: Hot spot at local (%d, %d)\n", 
               cart_rank, local_cx, local_cy);
    }
    
    /* ================================================================
     * STEP 6: Time-stepping loop with halo exchange
     * ================================================================ */
    
    double t_start = MPI_Wtime();
    
    for (int step = 0; step < NSTEPS; step++) {
        
        /* ----------------------------------------------------------
         * Halo exchange: exchange ghost zones with neighbors
         * ---------------------------------------------------------- */
        
        /* 
         * Y-direction exchange (rows): data is contiguous
         * Send my bottom row (j=1) to 'down', receive into top ghost (j=ny-1) from 'up'
         * Send my top row (j=ny_local) to 'up', receive into bottom ghost (j=0) from 'down'
         */
        
        /* Exchange with down/up neighbors */
        MPI_Sendrecv(&U(1, 1),        nx_local, MPI_DOUBLE, down, 0,
                     &U(1, ny-1),     nx_local, MPI_DOUBLE, up,   0,
                     cart_comm, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&U(1, ny_local), nx_local, MPI_DOUBLE, up,   1,
                     &U(1, 0),        nx_local, MPI_DOUBLE, down, 1,
                     cart_comm, MPI_STATUS_IGNORE);
        
        /* 
         * X-direction exchange (columns): data is strided, use derived type
         * Send my left column (i=1) to 'left', receive into right ghost from 'right'
         * Send my right column (i=nx_local) to 'right', receive into left ghost from 'left'
         */
        
        MPI_Sendrecv(&U(1, 1),        1, column_type, left,  2,
                     &U(nx-1, 1),     1, column_type, right, 2,
                     cart_comm, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&U(nx_local, 1), 1, column_type, right, 3,
                     &U(0, 1),        1, column_type, left,  3,
                     cart_comm, MPI_STATUS_IGNORE);
        
        /* ----------------------------------------------------------
         * Compute: 5-point stencil (2D Laplacian)
         * u_new = u + alpha * (u_left + u_right + u_down + u_up - 4*u)
         * ---------------------------------------------------------- */
        
        for (int i = 1; i <= nx_local; i++) {
            for (int j = 1; j <= ny_local; j++) {
                U_NEW(i, j) = U(i, j) + ALPHA * (
                    U(i-1, j) + U(i+1, j) +
                    U(i, j-1) + U(i, j+1) -
                    4.0 * U(i, j)
                );
            }
        }
        
        /* Swap pointers for next iteration */
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }
    
    double t_end = MPI_Wtime();
    
    /* ================================================================
     * STEP 7: Compute statistics and report
     * ================================================================ */
    
    /* Find local max temperature */
    double local_max = 0.0;
    for (int i = 1; i <= nx_local; i++) {
        for (int j = 1; j <= ny_local; j++) {
            if (U(i, j) > local_max) local_max = U(i, j);
        }
    }
    
    /* Global reduction to find overall max */
    double global_max;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    double elapsed = t_end - t_start;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    if (cart_rank == 0) {
        printf("\n=== Results ===\n");
        printf("Max temperature after %d steps: %.6f\n", NSTEPS, global_max);
        printf("Elapsed time: %.4f seconds\n", max_elapsed);
        printf("Grid points per second: %.2e\n", 
               (double)NX_GLOBAL * NY_GLOBAL * NSTEPS / max_elapsed);
    }
    
    /* ================================================================
     * STEP 8: Cleanup
     * ================================================================ */
    
    free(u);
    free(u_new);
    MPI_Type_free(&column_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}
