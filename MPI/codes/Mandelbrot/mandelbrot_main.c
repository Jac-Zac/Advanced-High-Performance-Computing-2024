/*
 * mandelbrot_main.c
 * 
 * Main driver for MPI Mandelbrot set calculation.
 * 
 * Usage: mpirun -np N ./mandelbrot_mpi [backend] [options]
 * 
 * Backends:
 *   lock   - Lock-based passive target (default)
 *   shmem  - Shared memory windows
 * 
 * Options:
 *   -w WIDTH      Image width (default: 1024)
 *   -h HEIGHT     Image height (default: 1024)
 *   -i MAXITER    Maximum iterations (default: 1000)
 *   -p PATCHSIZE  Initial patch size (default: 64)
 *   -o FILENAME   Output filename (default: mandelbrot.pgm)
 *   -xmin/-xmax/-ymin/-ymax  Complex plane bounds
 * 
 * Architecture:
 *   - Rank 0 (producer): initializes queue, collects results
 *   - Ranks 1..N-1 (consumers): fetch patches, compute, write back
 *   - If N=1, rank 0 does both roles
 */

#include "mandelbrot_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * Main program
 *============================================================================*/

int main(int argc, char **argv)
{
  {
    int provided;
    int ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if ( ret != MPI_SUCCESS ) {
      printf("An error occurred while initializing MPI library\n");
      return -1; }
    if ( provided < MPI_THREAD_FUNNELLED ) {
      printf("Provided MPI level is lower than required\n");
      return -2; }
  }
    
  int Rank, NRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &NRanks);
    
    /* Initialize context */
  Context ctx;
  init_context(&ctx, rank, num_procs);
    
  /* Select backend */
  QueueBackend *backend = &backend_lock;  /* Default */
  char *output_file = "mandelbrot.pgm";
    
  /* Parse arguments */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "lock") == 0) {
      backend = &backend_lock;
    } else if (strcmp(argv[i], "shmem") == 0) {
      backend = &backend_shmem;
    } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      output_file = argv[++i];
    }
  }
  parse_args(&ctx, argc, argv);
    
  if (rank == 0) {
    printf("=== MPI Mandelbrot Set Calculator ===\n");
    printf("Backend: %s\n", backend->name);
    printf("Processes: %d\n", num_procs);
    printf("Image: %d x %d\n", ctx.img_width, ctx.img_height);
    printf("Max iterations: %d\n", ctx.max_iter);
    printf("Initial patch size: %d\n", ctx.initial_patch_size);
    printf("Bounds: [%.3f, %.3f] x [%.3f, %.3f]\n",
	   ctx.bounds.x_min, ctx.bounds.x_max,
	   ctx.bounds.y_min, ctx.bounds.y_max);
    printf("Output: %s\n\n", output_file);
  }
    
    /* Initialize backend */
    double t_start = MPI_Wtime();
    
    int ret = backend->init(&ctx);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "Rank %d: Backend initialization failed\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double t_init = MPI_Wtime();
    
    /*========================================================================
     * Producer: Initialize queue with initial patches
     *========================================================================*/
    if (rank == PRODUCER_RANK) {
        printf("Producer: Initializing work queue...\n");
        
        int patch_size = ctx.initial_patch_size;
        int num_patches = 0;
        
        /* Tile the image with initial patches */
        for (int y = 0; y < ctx.img_height; y += patch_size) {
            for (int x = 0; x < ctx.img_width; x += patch_size) {
                Patch patch = { x, y, patch_size, 1 };
                backend->enqueue(&ctx, &patch);
                num_patches++;
            }
        }
        
        printf("Producer: Enqueued %d initial patches\n", num_patches);
    }
    
    /* Barrier to ensure queue is initialized before consumers start */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /*========================================================================
     * Consumer loop: All ranks (including producer) process patches
     * 
     * This implements work-stealing: each process independently fetches
     * patches until the queue is exhausted and no work is pending.
     *========================================================================*/
    double t_work_start = MPI_Wtime();
    int patches_processed = 0;
    
    while (1) {
        Patch patch;
        
        /* Try to get a patch from the queue */
        backend->dequeue(&ctx, &patch);
        
        if (patch.valid == TERMINATION_SIGNAL) {
            /* Queue exhausted and no pending work */
            break;
        }
        
        /* Process the patch */
        process_patch(&ctx, backend, &patch);
        patches_processed++;
    }
    
    double t_work_end = MPI_Wtime();
    
    /*========================================================================
     * Gather statistics and finalize
     *========================================================================*/
    
    /* Collect patch counts */
    int total_patches;
    MPI_Reduce(&patches_processed, &total_patches, 1, MPI_INT, MPI_SUM,
               PRODUCER_RANK, MPI_COMM_WORLD);
    
    /* Barrier before output */
    MPI_Barrier(MPI_COMM_WORLD);
    
    double t_final = MPI_Wtime();
    
    /* Producer writes output */
    if (rank == PRODUCER_RANK) {
        printf("\n=== Statistics ===\n");
        printf("Total patches processed: %d\n", total_patches);
        printf("Initialization time: %.3f s\n", t_init - t_start);
        printf("Computation time: %.3f s\n", t_work_end - t_work_start);
        printf("Total time: %.3f s\n", t_final - t_start);
        
        /* Per-process stats */
        printf("\nPer-rank breakdown:\n");
    }
    
    /* Print individual stats in order */
    for (int r = 0; r < num_procs; r++) {
        if (rank == r) {
            printf("  Rank %d: %d patches, %.3f s compute time\n",
                   rank, patches_processed, t_work_end - t_work_start);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    /* Write output file */
    if (rank == PRODUCER_RANK) {
        printf("\nWriting output to %s...\n", output_file);
        
        /* For shmem backend, ctx.image might be NULL after finalize,
         * so we need to copy before that */
        int *image_copy = malloc(ctx.img_width * ctx.img_height * sizeof(int));
        memcpy(image_copy, ctx.image, 
               ctx.img_width * ctx.img_height * sizeof(int));
        
        write_pgm(output_file, image_copy, ctx.img_width, ctx.img_height,
                  ctx.max_iter);
        free(image_copy);
        
        printf("Done!\n");
    }
    
    /* Cleanup */
    backend->finalize(&ctx);
    
    MPI_Finalize();
    return 0;
}
