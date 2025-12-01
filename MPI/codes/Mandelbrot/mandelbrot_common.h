/*
 * mandelbrot_common.h
 * 
 * Common definitions for MPI Mandelbrot set calculation
 * using producer-consumer paradigm with adaptive refinement.
 * 
 * The algorithm:
 *   1. Producer enqueues initial patches
 *   2. Consumers fetch patches from a shared queue (one-sided)
 *   3. For each patch: compute perimeter first
 *      - All in set → fill patch with max_iter
 *      - None in set → fill with average iteration
 *      - Mixed → subdivide into 4 sub-patches, re-enqueue
 *   4. Results written back to producer's image buffer (one-sided)
 */

#ifndef MANDELBROT_COMMON_H
#define MANDELBROT_COMMON_H

#include <mpi.h>
#include <stdint.h>
#include <stdbool.h>

/*============================================================================
 * Configuration constants
 *============================================================================*/

#define PRODUCER_RANK 0

/* Mandelbrot parameters */
#define DEFAULT_MAX_ITER    1000
#define DEFAULT_IMG_WIDTH   1024
#define DEFAULT_IMG_HEIGHT  1024
#define DEFAULT_PATCH_SIZE  64
#define MIN_PATCH_SIZE      4      /* Stop subdividing below this */

/* Queue parameters */
#define MAX_QUEUE_SIZE      65536  /* Maximum patches in queue */
#define TERMINATION_SIGNAL  -1     /* Special value to signal end */

/*============================================================================
 * Data structures
 *============================================================================*/

// in C++ you may want to define a "Bounds" template
// that can be instantiated for integer (pixels) or
// double (coordinates)

/*
 * Patch descriptor: defines a rectangular region to compute.
 * Stored in the shared queue.
 */
typedef struct {
  int x;          /* Bottom-left corner x (pixel coordinates)    */
  int y;          /* Bottom-left corner y (pixel coordinates)    */
  int xsize;      /* X Side length in pixels (square patches) */
  int ysize;      /* Y Side length in pixels (square patches) */
  int valid;      /* 1 if valid patch, 0 if empty slot, -1 if terminate */
} Patch;

/*
 * plane mapping for the Mandelbrot set
 */
typedef struct {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
} Bounds;

/*
 * Global computation context - shared by all modules
 */
typedef struct {
    /* MPI info */
  int rank;
  int nprocs;
  
  /* Image dimensions in pixels */
  unsigned int img_width;
  unsigned int img_height;
  
  /* Mandelbrot parameters */
  int max_iter;
  Bounds bounds;
  double r_xstep;  // xsize / n_x_pixels
  double r_ystep;  // xsize / n_y_pixels
  
  /* Initial patch size */
  int initial_patch_size;
  
  /* Image buffer (only allocated on producer) */
  int *image;
  
  /* Windows for one-sided communication */
  MPI_Win queue_win;      /* Queue window */
  MPI_Win image_win;      /* Image window for result writeback */
  MPI_Win counter_win;    /* Queue head/tail counters */
  
  /* Queue storage (on producer) */
  Patch *queue_buffer;
  int64_t *queue_counters; /* [0]=head (read), [1]=tail (write), [2]=pending */
  
} Context;

/*============================================================================
 * Queue operation function pointers (backend-specific)
 * 
 * These define the interface that each communication backend must implement.
 *============================================================================*/

typedef struct {
    /* Initialize the queue system */
    int (*init)(Context *ctx);
    
    /* Producer: enqueue a patch */
    int (*enqueue)(Context *ctx, const Patch *patch);
    
    /* Consumer: dequeue a patch (returns 0 on success, -1 if empty/done) */
    int (*dequeue)(Context *ctx, Patch *patch);
    
    /* Signal that a patch computation is complete (decrement pending) */
    int (*complete_patch)(Context *ctx);
    
    /* Check if all work is done */
    int (*is_done)(Context *ctx);
    
    /* Write computed patch results to image */
  int (*write_results)(Context *ctx, int x, int y, int xsize, int ysize,
                         const int *values);
    
    /* Cleanup */
    void (*finalize)(Context *ctx);
    
    /* Name for identification */
    const char *name;
    
} QueueBackend;

/*============================================================================
 * Mandelbrot computation functions (backend-agnostic)
 *============================================================================*/

/* Initialize context with default values */
void init_context(Context *ctx, int rank, int num_procs);

/* Parse command line arguments */
void parse_args(Context *ctx, int argc, char **argv);

/* Map pixel coordinates to complex plane */
static inline void pixel_to_point( double xmin, double xsize,
				   double ymin, double ysize,
				   double r_width, double r_heigth,
				   int px, int py, 
				   double *cx, double *cy )
{
  *cx = xmin + xsize * px * r_width;
  *cx = ymin + ysize * py * r_width;
}

/* Compute iteration count for a single point */
int iterate(double cx, double cy, int max_iter);

/* 
 * Compute perimeter of a patch.
 * Returns:
 *   1  if all points are IN the set (iterations == max_iter for all)
 *   0  if mixed (some in, some out)
 *  -1  if all points are OUT of the set
 * 
 * Also fills 'avg_iter' with average iteration count of perimeter.
 */
int compute_perimeter(const Context *ctx,
                      int x, int y,
		      int xsize, int ysize,
                      double *avg_iter);

/*
 * Process a single patch using the adaptive algorithm.
 * May enqueue sub-patches if refinement is needed.
 */
void process_patch(      Context      *ctx, 
                   const QueueBackend *backend,
                         Patch        *patch);

/* Write image to PGM file */
int write_pgm(const char *filename, const int *image, 
              int width, int height, int max_val);

/*============================================================================
 * Backend declarations
 *============================================================================*/

/* Each backend is defined in its own source file */
extern QueueBackend backend_pscw;
extern QueueBackend backend_lock;
extern QueueBackend backend_shmem;

#endif /* MANDELBROT_COMMON_H */
