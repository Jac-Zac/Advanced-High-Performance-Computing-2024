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


static int *patch_values = NULL;

/*============================================================================
 * Mandelbrot iteration
 *============================================================================*/

/*
 * Classic Mandelbrot iteration: z_{n+1} = z_n^2 + c
 * Returns the number of iterations before |z| > 2, or max_iter if bounded.
 */
int mandelbrot_iterate(double cx, double cy, int max_iter)
{
    double zx = 0.0, zy = 0.0;
    double zx2 = 0.0, zy2 = 0.0;
    int iter = 0;
    
    /* while loop exit when !z! > 4 */
    // this loop form is not vectorizable
    // do you want to import what we have seen
    // when discussin the vectorization ?
    while ( (zx2 + zy2 <= 4.0) && (iter < max_iter) )
      {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
      }
    
    return iter;
}

/*============================================================================
 * Perimeter computation
 *============================================================================*/

int compute_perimeter( int px0, int py0,         // the bottom-left corner, in pixels
		       int pxsize, int pysize,   // the pixel size
		       double x0, double y0,     // the top-left corner, in coordinates
		       double xstep,           
		       double ystep,
		       int max_iter;
                       double *avg_iter   )
{

  int  idx = 0;      
  int  all_in = 1;   /* All at max_iter? */
  int  any_in = 0;   /* Any at max_iter? */
  unsigned long long int sum = 0;

  double rx    = x0; // it is xmin + x * r_xstep;
  double ry    = y0; // it is ymin + y * r_ystep;
    
  
  /* bottom edge */
  for (int dpx = 0; dpx < pxsize; dpx++)
    {
      rx = x0 + dpx * xstep;
      int iter = mandelbrot_iterate(rx, ry, max_iter);
      
      sum += iter;

      any_in = (iter == max_iter);
      all_in = (iter < max_iter );
    }

  // rx here is at the last column
  // ry at the bottom row
  
  /* right edge */
  for (int dpy = 1; dpy < pysize; dpy++)
    {
      ry = y0 + dpy * ystep;
      int iter = mandelbrot_iterate(rx, ry, max_iter);
      
      sum += iter;
      
      any_in = (iter == max_iter);
      all_in = (iter < max_iter );
    }

  // rx here is at the last column
  // ry here is at the top row
  
  /* top edge */
  for (int dpx = pxsize; dpx >= 0; dpx--)
    {
      rx = x0 + dpx * xstep;
      int iter = mandelbrot_iterate(rx, ry, max_iter);
      
      sum += iter;
      
      any_in = (iter == max_iter);
      all_in = (iter < max_iter );
    }

  // rx here is at the first column
  // ry here is at the top row
  
  /* left edge */
  for (int dpy = pysize ; dpy < py0; dpy--)
    {
      ry = ymin + py * ystep;
      int iter = mandelbrot_iterate(rx, ry, max_iter);
      
      sum += iter;
      
      any_in = (iter == max_iter);
      all_in = (iter < max_iter );
    }
    
  *avg_iter = (idx > 0) ? (double)sum / idx : 0.0;
  
                              /* returns                                  */
  return (all_in - any_in);   /*  1 if all_in = 1 and any_in = 0  ALL IN  */
                              /* -1 if all_in = 0 and any_in = 1  MIXED   */
                              /*  0 if all_in = 0 and any_in = 0  ALL OUT */
}

/*============================================================================
 * Patch processing with adaptive refinement
 *============================================================================*/

void process_patch(       Context      *ctx, 
		    const QueueBackend *backend,
		          Patch        *patch )
{
  /* Allocate result buffer for patches */
  if ( patch_values == NULL )
    patch_values = malloc ( DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE * sizeof(int) );

  // ----------------------------------------------
  // patch boundaries
  //
  
  // top left corner, in pixels
  int px0    = patch->x;
  int py0    = patch->y;
  // size, in pixels
  int pxsize = patch->xsize;
  int pysize = patch->ysize;
  
  /* Clamp to image bounds */
  int actual_width  = (px0 + pxsize > ctx->img_width) ? ctx->img_width - px0 : pxsize;
  int actual_height = (py0 + pysize > ctx->img_height) ? ctx->img_height - py0 : pysize;
  
  if (actual_width <= 0 || actual_height <= 0) {
    backend->complete_patch(ctx);
    return;
  }

  int small_patch = (xsize <= MIN_PATCH_SIZE) && (ysize <= MIN_PATCH_SIZE);

  // --------------------------------------------
  // calculate perimeter
  //

  double r_xstep  = ctx->r_xstep;
  double r_ystep  = ctx->r_ystep;  
  double avg_iter = 0;
  int    status   = compute_perimeter(ctx, x, y, actual_width, actual_height, &avg_iter);

  if ( status >= 0 )
    {
      int iter_value = (status == 1 ? ctx->max_iter : (int)round(avg_iter) );
  
      for (int i = 0; i < actual_width * actual_height; i++)
	patch_values[i] = iter_value;

      backend->write_results(ctx, x, y, actual_width, actual_height, patch_values);
      backend->complete_patch(ctx);
    }
  
  /* Mixed */
  else if ( small_patch )
    {
      /* Too small to subdivide: compute every pixel */
      double cx, cy;
      for (int py = 0; py < actual_height; py++) {
	for (int px = 0; px < actual_width; px++) {
	  pixel_to_point(ctx, x + px, y + py, &cx, &cy);
	  patch_values[py * actual_width + px] = 
	    mandelbrot_iterate(cx, cy, ctx->max_iter);
	}
      }
      backend->write_results(ctx, x, y, actual_width, actual_height, patch_values);
      backend->complete_patch(ctx);
    }
  
  else
    {
      /* subdivide into 4 sub-patches */      
      int xhalf = xsize / 2;
      int yhalf = ysize / 2;
      Patch sub[4] = {
	{ x,         y,         xhalf, yhalf, 1 },   /* Top-left */
	{ x + xhalf, y,         xhalf, yhalf, 1 },   /* Top-right */
	{ x,         y + yhalf, xhalf, yhalf, 1 },   /* Bottom-left */
	{ x + xhalf, y + yhalf, xhalf, yhalf, 1 }    /* Bottom-right */
      };
      
      /* Enqueue all 4 sub-patches */
      for (int i = 0; i < 4; i++)
	{
	  /* Only enqueue if within image bounds */
	  if (sub[i].x < ctx->img_width && sub[i].y < ctx->img_height)
	    backend->enqueue(ctx, &sub[i]);	  
	}
            
      /* Original patch is done (but spawned children) */
      backend->complete_patch(ctx);

      return;
    }
  
}
