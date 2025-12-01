/*
 * backend_lock.c
 * 
 * Queue backend using MPI Passive (lock) synchronization.
 * 
 * 
 * For this implementation:
 *   - We use a polling model with short exposure epochs
 *   - Producer repeatedly posts windows for consumer access
 *   - Atomic operations used within epochs for queue manipulation
 */

#include "mandelbrot_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HEAD 0
#define TAIL 1
#define ACTV 2


/*============================================================================
 * Initialization
 *============================================================================*/

static int lock_init ( Context *ctx )
{
    
    /* Allocate and create windows */
    // there will be three windows:
    // - queued tasks  : the queue of patches to be calculated
    // - counter queue : defines the head and the tail of the task queue
    // - image         : the total image
    //
    MPI_Aint queue_size   = 0;
    MPI_Aint counter_size = 0;
    MPI_Aint image_size   = 0;
    
    if (ctx->rank == PRODUCER_RANK)
      {
	// NOTE: only the producer will actually
	//       allocate memory. All the other ranks
	//       will have zero-memory windows
	
        /* Queue buffer */
        queue_size        = MAX_QUEUE_SIZE * sizeof(Patch);
        //ctx->queue_buffer = malloc(queue_size);
        //memset(ctx->queue_buffer, 0, queue_size);
        
        /* Counters: [0]=head, [1]=tail, [2]=pending patches */
        counter_size = 3 * sizeof(int64_t);
        //ctx->queue_counters    = malloc(counter_size);
        ctx->queue_counters[HEAD] = 0;  /* head (next to dequeue) */
        ctx->queue_counters[TAIL] = 0;  /* tail (next to enqueue) */
        ctx->queue_counters[ACTV] = 0;  /* pending (active patches) */
        
        /* Image buffer */
        image_size = (MPI_Aint)ctx->img_width * ctx->img_height * sizeof(int);
        ctx->image = malloc(image_size);
        memset(ctx->image, 0, image_size);
    }

    int ret;
    
    /* allocate windows - all processes participate */
    ret = MPI_Win_create(queue_size, sizeof(Patch), MPI_INFO_NULL,
			 MPI_COMM_WORLD, ctx->queue_buffer, &ctx->queue_win);
    if (ret != MPI_SUCCESS) return ret;
    
    ret = MPI_Win_allocate(counter_size, sizeof(int64_t), MPI_INFO_NULL,
			   MPI_COMM_WORLD, ctx->queue_counters, &ctx->counter_win);
    if (ret != MPI_SUCCESS) return ret;
    
    ret = MPI_Win_allocate(image_size, sizeof(int), MPI_INFO_NULL,
			   MPI_COMM_WORLD, ctx->image, &ctx->image_win);
    if (ret != MPI_SUCCESS) return ret;

    if (ctx->rank == PRODUCER_RANK)
      {
        /* Queue buffer */
        memset(ctx->queue_buffer, 0, queue_size);
        
        /* Counters: [0]=head, [1]=tail, [2]=pending patches */
        ctx->queue_counters[0] = 0;  /* head (next to dequeue) */
        ctx->queue_counters[1] = 0;  /* tail (next to enqueue) */
        ctx->queue_counters[2] = 0;  /* pending (active patches) */
        
        /* Image buffer */
        memset(ctx->image, 0, image_size);
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_SUCCESS;
}

/*============================================================================
 * Queue operations using lock-based passive target
 *============================================================================*/

static int lock_enqueue(Context *ctx, const Patch *patch)
{
    int64_t tail;

    /*
     * Atomically reserve a slot using fetch-and-add on tail counter.
     * 
     * MPI_Fetch_and_op provides atomic read-modify-write:
     *   1. Reads current value at target
     *   2. Applies operation (SUM here) with provided value
     *   3. Returns original value
     * 
     * This is lock-free at the MPI level (uses hardware atomics if available).
     */
    
    /* Start access epoch to producer's counter window */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
    
    /* Fetch-and-add to get slot and increment tail atomically */
    MPI_Fetch_and_op(&(int64_t){1}, &tail, MPI_INT64_T, PRODUCER_RANK,
                     TAIL,
                     MPI_SUM, ctx->counter_win);
    
    /* Also increment pending counter */
    MPI_Accumulate(&(int64_t){1}, 1, MPI_INT64_T, PRODUCER_RANK,
                   ACTV,
                   1, MPI_INT64_T, MPI_SUM, ctx->counter_win);

    /* This will force the end of RMA too */
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    
    /* Insert the patch in the queue at the acquired slot */
    int slot = tail % MAX_QUEUE_SIZE;
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->queue_win);
    MPI_Put(patch, sizeof(Patch), MPI_BYTE, PRODUCER_RANK,
            slot, sizeof(Patch), MPI_BYTE, ctx->queue_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->queue_win);
    
    return MPI_SUCCESS;
}

static int lock_dequeue(Context *ctx, Patch *patch)
{
    int64_t head, tail;
    int success  = 0;
    int attempts = 0
    const int max_empty_attempts = 1000;  /* Prevent neverending spinning */
    
    while (!success)
      {

	// acquire a shared lock
        MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->counter_win);

	/* Read current head and tail */
        MPI_Get(&head, 1, MPI_INT64_T, PRODUCER_RANK, HEAD, 1, MPI_INT64_T,
                ctx->counter_win);
        MPI_Get(&tail, 1, MPI_INT64_T, PRODUCER_RANK, TAIL, 1, MPI_INT64_T,
                ctx->counter_win);
	MPI_Get(&pending, 1, MPI_INT64_T, PRODUCER_RANK, ACTV, 1, MPI_INT64_T,
		ctx->counter_win);
	
	// complete the ops
        MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
        
        if ( head >= tail )
	  {
	    if ( pending <= 0 )
	      {
		patch->valid = TERMINATION_SIGNAL;
		return 0;
	      }
	    
	    attempts++;
	    if ( ++attempts > max_empty_attempts ) {
	      attempts = 0; usleep(100); }
	    continue;
	  }
                    
	/* Try to claim a slot with compare-and-swap */
	int64_t new_head = head + 1;
	int64_t result;
        
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
        MPI_Compare_and_swap(&new_head, &head, &result, MPI_INT64_T,
                             PRODUCER_RANK, 0, ctx->counter_win);
        MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
        
        if (result == head)
	  {
            /* Successfully claimed slot 'head' */
            int slot = head % MAX_QUEUE_SIZE;
            
            MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->queue_win);
            MPI_Get(patch, sizeof(Patch), MPI_BYTE, PRODUCER_RANK,
                    slot, sizeof(Patch), MPI_BYTE, ctx->queue_win);
            MPI_Win_unlock(PRODUCER_RANK, ctx->queue_win);
            
            success = 1;
	  }
        /* else: CAS failed, retry */
      }
    
    return 0;
}

static int lock_complete_patch(Context *ctx)
{
    /* Decrement pending counter atomically */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
    MPI_Accumulate(&(int64_t){-1}, 1, MPI_INT64_T, PRODUCER_RANK,
                   ACTV, 1, MPI_INT64_T, MPI_SUM, ctx->counter_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    return MPI_SUCCESS;
}

static int lock_is_done(Context *ctx)
{
    int64_t head, tail, pending;
    
    MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->counter_win);
    MPI_Get(&head, 1, MPI_INT64_T, PRODUCER_RANK, 0, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Get(&tail, 1, MPI_INT64_T, PRODUCER_RANK, 1, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Get(&pending, 1, MPI_INT64_T, PRODUCER_RANK, 2, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    
    return (head >= tail && pending <= 0);
}

static int lock_write_results(Context *ctx, int x, int y, int size,
                              const int *values)
{
    /* 
     * Write computed values back to producer's image buffer.
     */
    
    int actual_width = (x + size > ctx->img_width) ? ctx->img_width - x : size;
    int actual_height = (y + size > ctx->img_height) ? ctx->img_height - y : size;
    
    /*
     * Write results row by row.
     * Using LOCK_EXCLUSIVE ensures no other process is writing to
     * overlapping memory (though our algorithm guarantees no overlap).
     */
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->image_win);
    
    /* Write row by row */
    for (int row = 0; row < actual_height; row++) {
        int img_offset = (y + row) * ctx->img_width + x;
        MPI_Put(&values[row * actual_width], actual_width, MPI_INT,
                PRODUCER_RANK, img_offset, actual_width, MPI_INT,
                ctx->image_win);
    }
    
    MPI_Win_unlock(PRODUCER_RANK, ctx->image_win);
    
    return MPI_SUCCESS;
}

static void lock_finalize(Context *ctx)
{
  /* Ensure all operations complete before freeing windows */
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (ctx->queue_win != MPI_WIN_NULL) MPI_Win_free(&ctx->queue_win);
  if (ctx->counter_win != MPI_WIN_NULL) MPI_Win_free(&ctx->counter_win);
  if (ctx->image_win != MPI_WIN_NULL) MPI_Win_free(&ctx->image_win);
  
  MPI_Group_free(&world_group);
  MPI_Group_free(&producer_group);
  MPI_Group_free(&consumer_group);
  
  if (ctx->rank == PRODUCER_RANK) {
    free(ctx->queue_buffer);
    free(ctx->queue_counters);
    /* image freed by caller */
  }
}

/*============================================================================
 * Backend interface
 *============================================================================*/

QueueBackend backend_lock = {
    .init           = lock_init,
    .enqueue        = lock_enqueue,
    .dequeue        = lock_dequeue,
    .complete_patch = lock_complete_patch,
    .is_done        = lock_is_done,
    .write_results  = lock_write_results,
    .finalize       = lock_finalize,
    .name           = "Lock-based" }
};
