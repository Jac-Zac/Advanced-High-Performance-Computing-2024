/*
 * backend_lock.c
 * 
 * Queue backend using MPI passive target synchronization with locks.
 * 
 * This uses MPI_Win_lock/unlock for fine-grained mutual exclusion:
 *   - MPI_LOCK_EXCLUSIVE: single writer access
 *   - MPI_LOCK_SHARED: multiple reader access
 * 
 * Key characteristics:
 *   - Most intuitive for dynamic producer-consumer patterns
 *   - Each operation is self-contained (no epoch coordination)
 *   - Lower latency for frequent small operations
 *   - Well-suited for work-stealing queues
 * 
 * Implementation notes:
 *   - Queue is a circular buffer with head/tail pointers
 *   - Compare-and-swap used for atomic dequeue
 *   - Fetch-and-add used for atomic enqueue
 */

#include "mandelbrot_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*============================================================================
 * Initialization
 *============================================================================*/

static int lock_init( Context *ctx)
{
    int ret;
    MPI_Aint queue_size = 0;
    MPI_Aint counter_size = 0;
    MPI_Aint image_size = 0;
    
    if (ctx->rank == PRODUCER_RANK) {
        /* Queue buffer */
        queue_size = MAX_QUEUE_SIZE * sizeof(Patch);
        ctx->queue_buffer = malloc(queue_size);
        memset(ctx->queue_buffer, 0, queue_size);
        
        /* Counters: [0]=head, [1]=tail, [2]=pending patches */
        counter_size = 3 * sizeof(int64_t);
        ctx->queue_counters = malloc(counter_size);
        ctx->queue_counters[0] = 0;  /* head (next to dequeue) */
        ctx->queue_counters[1] = 0;  /* tail (next to enqueue) */
        ctx->queue_counters[2] = 0;  /* pending work count */
        
        /* Image buffer */
        image_size = (MPI_Aint)ctx->img_width * ctx->img_height * sizeof(int);
        ctx->image = malloc(image_size);
        memset(ctx->image, 0, image_size);
    }
    
    /* 
     * Create windows with MPI_INFO hints for optimization.
     * These hints tell the MPI implementation we'll use passive target.
     */
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none");  /* Allow reordering */
    MPI_Info_set(info, "accumulate_ops", "same_op_no_op"); /* Optimization hint */
    
    ret = MPI_Win_create(ctx->queue_buffer, queue_size, sizeof(Patch),
                         info, MPI_COMM_WORLD, &ctx->queue_win);
    if (ret != MPI_SUCCESS) goto cleanup;
    
    ret = MPI_Win_create(ctx->queue_counters, counter_size, sizeof(int64_t),
                         info, MPI_COMM_WORLD, &ctx->counter_win);
    if (ret != MPI_SUCCESS) goto cleanup;
    
    ret = MPI_Win_create(ctx->image, image_size, sizeof(int),
                         info, MPI_COMM_WORLD, &ctx->image_win);
    if (ret != MPI_SUCCESS) goto cleanup;
    
    MPI_Info_free(&info);
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_SUCCESS;
    
cleanup:
    MPI_Info_free(&info);
    return ret;
}

/*============================================================================
 * Queue operations using lock-based passive target
 *============================================================================*/

static int lock_enqueue( Context *ctx, const Patch *patch)
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
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
    
    /* Increment tail, get old value (our slot) */
    MPI_Fetch_and_op(&(int64_t){1}, &tail, MPI_INT64_T, PRODUCER_RANK,
                     1, /* offset 1 = tail */
                     MPI_SUM, ctx->counter_win);
    
    /* Also increment pending counter */
    MPI_Accumulate(&(int64_t){1}, 1, MPI_INT64_T, PRODUCER_RANK,
                   2, /* offset 2 = pending */
                   1, MPI_INT64_T, MPI_SUM, ctx->counter_win);
    
    /* Flush to ensure operations complete before unlock */
    MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    
    /* Write patch data to our reserved slot */
    int slot = tail % MAX_QUEUE_SIZE;
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->queue_win);
    MPI_Put(patch, sizeof(Patch), MPI_BYTE, PRODUCER_RANK,
            slot, sizeof(Patch), MPI_BYTE, ctx->queue_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->queue_win);
    
    return MPI_SUCCESS;
}

static int lock_dequeue(MandelbrotContext *ctx, Patch *patch)
{
    int64_t head, tail;
    int attempts = 0;
    const int max_empty_attempts = 1000;  /* Prevent infinite spinning */
    
    while (1) {
        /*
         * Read current head and tail with shared lock.
         * Shared locks allow concurrent reads but block exclusive.
         */
        MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->counter_win);
        MPI_Get(&head, 1, MPI_INT64_T, PRODUCER_RANK, 0, 1, MPI_INT64_T,
                ctx->counter_win);
        MPI_Get(&tail, 1, MPI_INT64_T, PRODUCER_RANK, 1, 1, MPI_INT64_T,
                ctx->counter_win);
        MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
        MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
        
        if (head >= tail) {
            /* Queue appears empty - check termination condition */
            int64_t pending;
            MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->counter_win);
            MPI_Get(&pending, 1, MPI_INT64_T, PRODUCER_RANK, 2, 1, MPI_INT64_T,
                    ctx->counter_win);
            MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
            MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
            
            if (pending <= 0) {
                /* No pending work: termination */
                patch->valid = TERMINATION_SIGNAL;
                return 0;
            }
            
            /* Work still pending somewhere - spin wait */
            attempts++;
            if (attempts > max_empty_attempts) {
                /* Yield to reduce contention */
                usleep(100);
                attempts = 0;
            }
            continue;
        }
        
        /*
         * Try to claim slot 'head' using compare-and-swap.
         * 
         * MPI_Compare_and_swap atomically:
         *   1. Compares target value with 'compare' parameter
         *   2. If equal, replaces with 'origin' parameter
         *   3. Returns original target value in 'result'
         * 
         * This enables lock-free claiming of queue slots.
         */
        int64_t new_head = head + 1;
        int64_t result;
        
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
        MPI_Compare_and_swap(&new_head, &head, &result, MPI_INT64_T,
                             PRODUCER_RANK, 0, ctx->counter_win);
        MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
        MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
        
        if (result == head) {
            /* CAS succeeded - we own slot 'head' */
            int slot = head % MAX_QUEUE_SIZE;
            
            MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->queue_win);
            MPI_Get(patch, sizeof(Patch), MPI_BYTE, PRODUCER_RANK,
                    slot, sizeof(Patch), MPI_BYTE, ctx->queue_win);
            MPI_Win_unlock(PRODUCER_RANK, ctx->queue_win);
            
            return 0;
        }
        /* CAS failed: another consumer claimed it, retry */
    }
}

static int lock_complete_patch(MandelbrotContext *ctx)
{
    /* Atomically decrement pending counter */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->counter_win);
    MPI_Accumulate(&(int64_t){-1}, 1, MPI_INT64_T, PRODUCER_RANK,
                   2, 1, MPI_INT64_T, MPI_SUM, ctx->counter_win);
    MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    return MPI_SUCCESS;
}

static int lock_is_done(MandelbrotContext *ctx)
{
    int64_t head, tail, pending;
    
    MPI_Win_lock(MPI_LOCK_SHARED, PRODUCER_RANK, 0, ctx->counter_win);
    MPI_Get(&head, 1, MPI_INT64_T, PRODUCER_RANK, 0, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Get(&tail, 1, MPI_INT64_T, PRODUCER_RANK, 1, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Get(&pending, 1, MPI_INT64_T, PRODUCER_RANK, 2, 1, MPI_INT64_T,
            ctx->counter_win);
    MPI_Win_flush(PRODUCER_RANK, ctx->counter_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->counter_win);
    
    return (head >= tail && pending <= 0);
}

static int lock_write_results(MandelbrotContext *ctx, int x, int y, int size,
                              const int *values)
{
    int actual_width = (x + size > ctx->img_width) ? ctx->img_width - x : size;
    int actual_height = (y + size > ctx->img_height) ? ctx->img_height - y : size;
    
    /*
     * Write results row by row.
     * Using LOCK_EXCLUSIVE ensures no other process is writing to
     * overlapping memory (though our algorithm guarantees no overlap).
     */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, PRODUCER_RANK, 0, ctx->image_win);
    
    for (int row = 0; row < actual_height; row++) {
        int img_offset = (y + row) * ctx->img_width + x;
        MPI_Put(&values[row * actual_width], actual_width, MPI_INT,
                PRODUCER_RANK, img_offset, actual_width, MPI_INT,
                ctx->image_win);
    }
    
    /*
     * MPI_Win_flush ensures all operations complete before unlock.
     * This is important for correctness when using passive target.
     */
    MPI_Win_flush(PRODUCER_RANK, ctx->image_win);
    MPI_Win_unlock(PRODUCER_RANK, ctx->image_win);
    
    return MPI_SUCCESS;
}

static void lock_finalize(MandelbrotContext *ctx)
{
    /* Ensure all operations complete before freeing windows */
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (ctx->queue_win != MPI_WIN_NULL) MPI_Win_free(&ctx->queue_win);
    if (ctx->counter_win != MPI_WIN_NULL) MPI_Win_free(&ctx->counter_win);
    if (ctx->image_win != MPI_WIN_NULL) MPI_Win_free(&ctx->image_win);
    
    if (ctx->rank == PRODUCER_RANK) {
        free(ctx->queue_buffer);
        free(ctx->queue_counters);
        /* image freed by caller if needed */
    }
}

/*============================================================================
 * Backend interface
 *============================================================================*/

QueueBackend backend_lock = {
    .init = lock_init,
    .enqueue = lock_enqueue,
    .dequeue = lock_dequeue,
    .complete_patch = lock_complete_patch,
    .is_done = lock_is_done,
    .write_results = lock_write_results,
    .finalize = lock_finalize,
    .name = "Lock-based (Passive Target)"
};
