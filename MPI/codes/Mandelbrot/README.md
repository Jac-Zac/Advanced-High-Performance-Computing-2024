---

## Critical Design Notes

A few things you should be aware of before using this with students:

### 1. Termination Detection is Tricky

The `pending` counter approach works but has a subtle race: between checking `head >= tail` and checking `pending`, a new patch could be enqueued. The code handles this with retry loops, but it's worth discussing. Alternative approaches include explicit termination messages or barrier-based epoch completion.

### 3. The Shared Memory Backend Assumes Single-Node

`MPI_Win_allocate_shared` only works within `MPI_COMM_TYPE_SHARED` groups (typically one node). For multi-node, you'd need a hybrid: shmem within node, RMA between nodes. For the sake of simpliciy this code warns about this and stops.

### 4. Potential Race in `shmem_write_results`

The direct `memcpy` without MPI synchronization relies on the guarantee that patches don't overlap — which is true by construction. Pay attention to that.

### 5. Busy-Wait Spinning

The dequeue loops spin with `usleep(100)` when the queue is empty. For production, we may opt for something more sophisticated (possibly `MPI_Win_lock_all` with `MPI_Win_sync` for progress, or hybrid with message-based wakeup).

### Exercise

Can you develop a fence-based backend as well?
It's often the simplest synchronization and makes a good "baseline" for comparison.
