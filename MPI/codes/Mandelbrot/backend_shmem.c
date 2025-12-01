/*
 * backend_shmem.c
 * 
 * Queue backend using MPI shared memory windows (MPI_Win_allocate_shared).
 * 
 * This exploits the MPI-3 shared memory model for processes on the same node:
 *   - MPI_Win_allocate_shared creates a window in shared memory
 *   - MPI_Win_shared_query retrieves direct pointers to remote memory
 *   - Direct load/store access (no MPI calls) for intra-node communication
 *   - Falls back to RMA for inter-node (if using multiple nodes)
 * 
 * Key characteristics:
 *   - Lowest latency for intra-node communication
 *   - Direct pointer access after setup
 *   - Requires explicit memory barriers for synchronization
 *   - Natural fit for shared-memory parallel patterns
 * 
 * Implementation notes:
 *   - We create a shared memory communicator for processes on same node
 *   - Within node: direct pointer access with atomics
 *   - Between nodes: would need additional RMA layer (not implemented here)
 *   - For single-node runs, this is the most efficient approach
 * 
 * IMPORTANT: This backend assumes all processes are on the SAME node.
 * For multi-node, you'd need hybrid approach (shmem within node, RMA between).
 */

#include "mandelbrot_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

/*============================================================================
 * Local state for shared memory backend
 *============================================================================*/

/* Communicator for processes sharing memory */
static MPI_Comm shmem_comm = MPI_COMM_NULL;
static int shmem_rank;
static int shmem_size;

/* Direct pointers to shared memory regions */
static Patch           *queue_ptr    = NULL;
static _Atomic int64_t *counters_ptr = NULL;  /* Using C11 atomics */
static         int     *image_ptr    = NULL;

/*============================================================================
 * Initialization
 *============================================================================*/

static int shmem_init ( Context *ctx )
{
    int ret;
    
    /*
     * MPI_Comm_split_type creates a communicator for processes
     * that share a specific resource. MPI_COMM_TYPE_SHARED groups
     * processes that can share memory (typically same node).
     */
    ret = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                              MPI_INFO_NULL, &shmem_comm);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "Failed to create shared memory communicator\n");
        return ret;
    }
    
    MPI_Comm_rank(shmem_comm, &shmem_rank);
    MPI_Comm_size(shmem_comm, &shmem_size);
    
    /* Verify all processes are in shared memory domain */
    /*
     * NOTE: normally, you do not want that all the processes
     *       stay in the same node; you build bridges among
     *       nodes via a communicator hierarchy.
     *       In this case we simplify.
     */
    if (shmem_size != ctx->num_procs) {
        if (ctx->rank == 0) {
            fprintf(stderr, "Warning: Not all processes share memory.\n"
                    "  World size: %d, Shared memory size: %d\n"
                    "  This backend requires single-node execution.\n",
                    ctx->num_procs, shmem_size);
	    return -1;
        }
    }
    
    /*
     * MPI_Win_allocate_shared allocates memory that is directly
     * addressable by all processes in the communicator.
     * 
     * Only rank 0 in the shared communicator allocates; others
     * pass size=0 and query for the pointer.
     */
    
    MPI_Aint queue_size = 0;
    MPI_Aint counter_size = 0;
    MPI_Aint image_size = 0;
    
    if (shmem_rank == 0) {
        queue_size = MAX_QUEUE_SIZE * sizeof(Patch);
        counter_size = 3 * sizeof(int64_t);  /* head, tail, pending */
        image_size = (MPI_Aint)ctx->img_width * ctx->img_height * sizeof(int);
    }
    
    /* Create shared memory windows */
    void *base_ptr;
    
    /* Queue window */
    ret = MPI_Win_allocate_shared(queue_size, sizeof(Patch), MPI_INFO_NULL,
                                  shmem_comm, &base_ptr, &ctx->queue_win);
    if (ret != MPI_SUCCESS) return ret;
    
    /* Query for actual pointer (works from any rank) */
    MPI_Aint size;
    int disp_unit;
    MPI_Win_shared_query(ctx->queue_win, 0, &size, &disp_unit, &queue_ptr);
    
    /* Counter window */
    ret = MPI_Win_allocate_shared(counter_size, sizeof(int64_t), MPI_INFO_NULL,
                                  shmem_comm, &base_ptr, &ctx->counter_win);
    if (ret != MPI_SUCCESS) return ret;
    
    MPI_Win_shared_query(ctx->counter_win, 0, &size, &disp_unit, 
                         (void**)&counters_ptr);
    
    /* Image window */
    ret = MPI_Win_allocate_shared(image_size, sizeof(int), MPI_INFO_NULL,
                                  shmem_comm, &base_ptr, &ctx->image_win);
    if (ret != MPI_SUCCESS) return ret;
    
    MPI_Win_shared_query(ctx->image_win, 0, &size, &disp_unit, &image_ptr);
    
    /* Initialize on rank 0 */
    if (shmem_rank == 0) {
        memset(queue_ptr, 0, MAX_QUEUE_SIZE * sizeof(Patch));
        atomic_store(&counters_ptr[0], 0);  /* head */
        atomic_store(&counters_ptr[1], 0);  /* tail */
        atomic_store(&counters_ptr[2], 0);  /* pending */
        memset(image_ptr, 0, ctx->img_width * ctx->img_height * sizeof(int));
        
        /* Store pointer for producer access */
        ctx->image = image_ptr;
        ctx->queue_buffer = queue_ptr;
        ctx->queue_counters = (int64_t*)counters_ptr;
    }
    
    /* Memory barrier to ensure initialization is visible */
    MPI_Barrier(shmem_comm);
    
    return MPI_SUCCESS;
}

/*============================================================================
 * Queue operations using direct shared memory access
 * 
 * Since we have direct pointers, we use C11 atomics for synchronization.
 * This provides lock-free operations with hardware memory ordering.
 *============================================================================*/

static int shmem_enqueue( Context *ctx, const Patch *patch)
{
    /*
     * Atomically reserve a slot using fetch_add on tail.
     * atomic_fetch_add returns the OLD value, which is our slot.
     */
    int64_t tail = atomic_fetch_add(&counters_ptr[1], 1);
    
    /* Also increment pending */
    atomic_fetch_add(&counters_ptr[2], 1);
    
    /* Write patch to our slot */
    int slot = tail % MAX_QUEUE_SIZE;
    
    /*
     * Memory fence before store to ensure the slot index is
     * established before we write data.
     */
    atomic_thread_fence(memory_order_release);
    
    /* Direct store to shared memory */
    queue_ptr[slot] = *patch;
    
    /* Ensure write is visible */
    atomic_thread_fence(memory_order_release);
    
    return MPI_SUCCESS;
}

static int shmem_dequeue( Context *ctx, Patch *patch)
{
  (void)ctx;  // we use atomics; let's avoid the warning "unused argument"
  
  int attempts = 0;
  
  while (1) {
    /* Read current head and tail with acquire semantics */
    int64_t head = atomic_load_explicit(&counters_ptr[0], 
					memory_order_acquire);
    int64_t tail = atomic_load_explicit(&counters_ptr[1], 
					memory_order_acquire);
    
    if (head >= tail) {
      /* Queue empty - check termination */
      int64_t pending = atomic_load_explicit(&counters_ptr[2],
					     memory_order_acquire);
      if (pending <= 0) {
	patch->valid = TERMINATION_SIGNAL;
	return 0;
      }
      
      /* Spin wait */
      attempts++;
      if (attempts > 1000) {
	usleep(10);  /* Shorter sleep - shmem is fast */
	attempts = 0;
      }
      continue;
    }
    
    /*
     * Try to claim slot using compare_exchange (CAS).
     * This atomically: if head == expected, set head = desired
     */
    int64_t expected = head;
    int64_t desired = head + 1;
    
    if (atomic_compare_exchange_strong(&counters_ptr[0], 
				       &expected, desired)) {
      /* CAS succeeded - we own slot 'head' */
      int slot = head % MAX_QUEUE_SIZE;
      
      /* Memory fence before reading data */
      atomic_thread_fence(memory_order_acquire);
      
      /* Direct load from shared memory */
      *patch = queue_ptr[slot];
      
      return 0;
    }
    /* CAS failed, retry */
  }
}

static int shmem_complete_patch ( Context *ctx )
{
  (void)ctx;  // we use atomics; let's avoid the warning "unused argument"
  /* Atomically decrement pending */
  atomic_fetch_sub(&counters_ptr[2], 1);
  return MPI_SUCCESS;
}

static int shmem_is_done( Context *ctx )
{
  (void)ctx;  // we use atomics; let's avoid the warning "unused argument"
  int64_t head    = atomic_load_explicit(&counters_ptr[0], memory_order_acquire);
  int64_t tail    = atomic_load_explicit(&counters_ptr[1], memory_order_acquire);
  int64_t pending = atomic_load_explicit(&counters_ptr[2], memory_order_acquire);
  
  return (head >= tail && pending <= 0);
}

static int shmem_write_results( Context *ctx, int x, int y, int size,
                                const int *values)
{
  int img_width  = ctx->img_width;
  int img_height = ctx->img_heigth;
  
  int actual_width = (x + size > img_width) ? img_width - x : size;
  int actual_height = (y + size > img_height) ? img_height - y : size;
  
  /*
   * Direct memory copy to shared image buffer.
   * No locking needed since patches don't overlap.
   * 
   * Memory fence ensures writes are visible to other processes.
   */
  for (int row = 0; row < actual_height; row++) {
    int img_offset = (y + row) * img_width + x;
    memcpy(&image_ptr[img_offset], &values[row * actual_width],
	   actual_width * sizeof(int));
  }
  
  /* Ensure writes are visible */
  atomic_thread_fence(memory_order_release);
  
  return MPI_SUCCESS;
}

static void shmem_finalize( Context *ctx )
{
  /* Barrier to ensure all operations complete */
  MPI_Barrier(shmem_comm);
  
  /* 
   * Free shared windows.
   * MPI_Win_free for shared windows also frees the underlying memory.
   */
  if (ctx->queue_win != MPI_WIN_NULL) MPI_Win_free(&ctx->queue_win);
  if (ctx->counter_win != MPI_WIN_NULL) MPI_Win_free(&ctx->counter_win);
  if (ctx->image_win != MPI_WIN_NULL) MPI_Win_free(&ctx->image_win);
  
  if (shmem_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&shmem_comm);
  }
  
  /* Clear pointers (memory freed by MPI_Win_free) */
  queue_ptr = NULL;
  counters_ptr = NULL;
  image_ptr = NULL;
  
  /* 
   * Don't free ctx->image etc. on producer - 
   * they point to shared memory freed above 
     */
  if (ctx->rank == PRODUCER_RANK) {
    ctx->image = NULL;
    ctx->queue_buffer = NULL;
    ctx->queue_counters = NULL;
  }
}

/*============================================================================
 * Backend interface
 *============================================================================*/

QueueBackend backend_shmem = {
    .init = shmem_init,
    .enqueue = shmem_enqueue,
    .dequeue = shmem_dequeue,
    .complete_patch = shmem_complete_patch,
    .is_done = shmem_is_done,
    .write_results = shmem_write_results,
    .finalize = shmem_finalize,
    .name = "Shared Memory Windows "
};
