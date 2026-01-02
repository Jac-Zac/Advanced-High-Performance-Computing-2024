
/* ────────────────────────────────────────────────────────────────────────── *
 │                                                                            │
 │ This file is part of the exercises for the Lectures on                     │
 │   "Foundations of High Performance Computing"                              │
 │ given at                                                                   │
 │   Master in HPC and                                                        │
 │   Master in Data Science and Scientific Computing                          │
 │ @ SISSA, ICTP and University of Trieste                                    │
 │                                                                            │
 │ contact: luca.tornatore@inaf.it                                            │
 │                                                                            │
 │     This is free software; you can redistribute it and/or modify           │
 │     it under the terms of the GNU General Public License as published by   │
 │     the Free Software Foundation; either version 3 of the License, or      │
 │     (at your option) any later version.                                    │
 │     This code is distributed in the hope that it will be useful,           │
 │     but WITHOUT ANY WARRANTY; without even the implied warranty of         │
 │     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          │
 │     GNU General Public License for more details.                           │
 │                                                                            │
 │     You should have received a copy of the GNU General Public License      │
 │     along with this program.  If not, see <http://www.gnu.org/licenses/>   │
 │                                                                            │
 * ────────────────────────────────────────────────────────────────────────── */

#if defined(__STDC__)
#if (__STDC_VERSION__ >= 199901L)
#define _XOPEN_SOURCE 700
#endif
#endif
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {

#pragma omp parallel
  {
    int me = omp_get_thread_num();

#pragma omp single nowait
    {
      printf(" »Yuk yuk, here is thread %d from "
             "within the single region\n",
             omp_get_thread_num());

#pragma omp task
      {
        printf("\tHi, here is thread %d "
               "running task A\n",
               omp_get_thread_num());
      }

     #pragma omp taskwait
      fflush(stdout);
      printf(" «Yuk yuk, it is still me, thread %d "
             "inside single region after all tasks ended\n",
             me);
    }

    // WARNING: Will wait for the task inside the {} so this has no
    // corresponding task so using it or commented is the same thing
#pragma omp taskwait

#pragma omp barrier

    //#pragma omp barrier
    
    printf(" :Hi, here is thread %d after the end "
           "of the single region, I'm stuck waiting "
           "all the others\n",
           omp_get_thread_num());
  }

  return 0;
}
