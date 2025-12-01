/*
 * mandelbrot_core.c
 * 
 * Core Mandelbrot set computation routines.
 * These are independent of the communication backend.
 */

#include "mandelbrot_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * Context initialization
 *============================================================================*/

void init_context(Context *ctx, int rank, int nprocs)
{
  memset(ctx, 0, sizeof(*ctx));
  
  ctx->rank = rank;
  ctx->nprocs = nprocs;
  
  /* Default image dimensions */
  ctx->img_width = DEFAULT_IMG_WIDTH;
  ctx->img_height = DEFAULT_IMG_HEIGHT;
  
  /* Default Mandelbrot parameters - classic view */
  ctx->max_iter = DEFAULT_MAX_ITER;
  ctx->bounds.x_min = -2.0;
  ctx->bounds.x_max = 1.0;
  ctx->bounds.y_min = -1.5;
  ctx->bounds.y_max = 1.5;
  
  ctx->initial_patch_size = DEFAULT_PATCH_SIZE;
  
  /* Windows initialized to MPI_WIN_NULL */
  ctx->queue_win = MPI_WIN_NULL;
  ctx->image_win = MPI_WIN_NULL;
  ctx->counter_win = MPI_WIN_NULL;
  
  return;
}

void mandelbrot_parse_args(Context *ctx, int argc, char **argv)
{
  for (int i = 1; i < argc; i++)
    {
      if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
	ctx->img_width = atoi(argv[++i]);
      } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
	ctx->img_height = atoi(argv[++i]);
      } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
	ctx->max_iter = atoi(argv[++i]);
      } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
	ctx->initial_patch_size = atoi(argv[++i]);
      } else if (strcmp(argv[i], "-xmin") == 0 && i + 1 < argc) {
	ctx->bounds.x_min = atof(argv[++i]);
      } else if (strcmp(argv[i], "-xmax") == 0 && i + 1 < argc) {
	ctx->bounds.x_max = atof(argv[++i]);
      } else if (strcmp(argv[i], "-ymin") == 0 && i + 1 < argc) {
	ctx->bounds.y_min = atof(argv[++i]);
      } else if (strcmp(argv[i], "-ymax") == 0 && i + 1 < argc) {
	ctx->bounds.y_max = atof(argv[++i]);
      }
    }
  return;
}


/*============================================================================
 * Output
 *============================================================================*/

int write_pgm(const char *filename, const int *image, 
              int width, int height, int max_val)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open output file");
        return -1;
    }
    
    /* PGM header */
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    
    /* Convert iterations to grayscale and write */
    unsigned char *row = malloc(width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int iter = image[y * width + x];
            /* Color mapping: in set (max_iter) = black, outside = grayscale */
            if (iter >= max_val) {
                row[x] = 0;  /* In set: black */
            } else {
                /* Logarithmic scaling for better contrast */
                double t = (double)iter / max_val;
                row[x] = (unsigned char)(255 * sqrt(t));
            }
        }
        fwrite(row, 1, width, fp);
    }
    
    free(row);
    fclose(fp);
    return 0;
}
