"""
advanced_parallel_concepts.py
Implementation of advanced parallelization concepts
Based on teacher's examples (Partitioning, Mapping, Agglomeration)

Task 3: Parallel Matrix Multiplication
Big Data - University of Las Palmas de Gran Canaria
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple, List


# ==================== HELPER FUNCTIONS (OUTSIDE THE CLASS) ====================
# Necessary for Windows multiprocessing

def _compute_partition_worker(args):
    """Worker for partitioning - must be at module level for Windows"""
    partition_id, start_row, end_row, A, B = args
    m, p = A.shape[1], B.shape[1]
    
    print(f"   Partition {partition_id}: rows {start_row} to {end_row-1}")
    
    result = np.zeros((end_row - start_row, p))
    for i in range(end_row - start_row):
        for j in range(p):
            for k in range(m):
                result[i, j] += A[start_row + i, k] * B[k, j]
    
    return (start_row, result)


def _compute_agglomerated_block_worker(args):
    """Worker for agglomeration - must be at module level for Windows"""
    block_id, start_row, end_row, A, B = args
    m, p = A.shape[1], B.shape[1]
    
    print(f"   Block-{block_id}: processes {end_row - start_row} rows")
    
    result = np.zeros((end_row - start_row, p))
    for i in range(end_row - start_row):
        for j in range(p):
            for k in range(m):
                result[i, j] += A[start_row + i, k] * B[k, j]
    
    return (start_row, result)


class AdvancedParallelConcepts:
    """
    Implementations demonstrating advanced parallelization concepts:
    - Partitioning (Data partitioning)
    - Mapping (Task mapping)
    - Agglomeration (Task agglomeration)
    
    Based on teacher's Java examples but adapted to Python
    """
    
    def __init__(self, num_cores: int = None):
        self.num_cores = num_cores or mp.cpu_count()
    
    # ==================== PARTITIONING ====================
    
    def partitioning_multiplication(self, A: np.ndarray, B: np.ndarray, 
                                   num_partitions: int = None) -> np.ndarray:
        """
        PARTITIONING: Divides data into independent blocks
        
        Concept: Similar to PartitioningExample.java
        - Divides matrix A into horizontal partitions
        - Each partition computes its result independently
        - Results are combined at the end
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            num_partitions: Number of partitions (blocks)
        
        Returns:
            Result matrix nxp
        """
        if num_partitions is None:
            num_partitions = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Incompatible dimensions"
        
        C = np.zeros((n, p))
        
        # Calculate partition size
        partition_size = int(np.ceil(n / num_partitions))
        
        print(f"\n📦 PARTITIONING:")
        print(f"   Total rows: {n}")
        print(f"   Partitions: {num_partitions}")
        print(f"   Rows per partition: {partition_size}")
        
        # Prepare arguments for workers
        tasks = []
        for part_id in range(num_partitions):
            start_row = part_id * partition_size
            end_row = min(start_row + partition_size, n)
            tasks.append((part_id, start_row, end_row, A, B))
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=num_partitions) as executor:
            results = executor.map(_compute_partition_worker, tasks)
            
            # Collect results
            for start_row, result in results:
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Partitioning completed")
        return C
    
    # ==================== MAPPING ====================
    
    def mapping_multiplication(self, A: np.ndarray, B: np.ndarray,
                              num_threads: int = None) -> np.ndarray:
        """
        MAPPING: Assigns specific tasks to specific threads/processes
        
        Concept: Similar to MappingExample.java
        - Each thread has a unique ID and assigned work block
        - Explicitly maps which thread processes which data
        - Useful when we want control over task assignment
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            num_threads: Number of threads to map
        
        Returns:
            Result matrix nxp
        """
        if num_threads is None:
            num_threads = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Incompatible dimensions"
        
        C = np.zeros((n, p))
        block_size = int(np.ceil(n / num_threads))
        
        print(f"\n🗺️ MAPPING:")
        print(f"   Available threads: {num_threads}")
        print(f"   Block per thread: {block_size} rows")
        
        def mapped_task(thread_id: int) -> Tuple[int, int, np.ndarray]:
            """Task mapped to a specific thread"""
            start_row = thread_id * block_size
            end_row = min(start_row + block_size, n)
            
            print(f"   Thread-{thread_id} → rows [{start_row}:{end_row-1}]")
            
            result = np.zeros((end_row - start_row, p))
            for i in range(end_row - start_row):
                for j in range(p):
                    for k in range(m):
                        result[i, j] += A[start_row + i, k] * B[k, j]
            
            return (thread_id, start_row, result)
        
        # Explicitly map tasks to threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mapped_task, tid) for tid in range(num_threads)]
            
            for future in as_completed(futures):
                thread_id, start_row, result = future.result()
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Mapping completed")
        return C
    
    # ==================== AGGLOMERATION ====================
    
    def agglomeration_multiplication(self, A: np.ndarray, B: np.ndarray,
                                    num_agglomerations: int = None) -> np.ndarray:
        """
        AGGLOMERATION: Groups small tasks into larger tasks
        
        Concept: Similar to AgglomerationExample.java
        - Instead of creating a task per row, we group multiple rows
        - Reduces overhead of task creation/management
        - Balance between parallelism and overhead
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            num_agglomerations: Number of large groups (< num_rows)
        
        Returns:
            Result matrix nxp
        """
        if num_agglomerations is None:
            # By default: agglomeration = cores/2 (fewer tasks, more work each one)
            num_agglomerations = max(1, self.num_cores // 2)
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Incompatible dimensions"
        
        C = np.zeros((n, p))
        
        # Large size of each agglomeration
        agglomeration_size = int(np.ceil(n / num_agglomerations))
        
        print(f"\n🔗 AGGLOMERATION:")
        print(f"   Total rows: {n}")
        print(f"   Agglomerations: {num_agglomerations}")
        print(f"   Rows per agglomeration: {agglomeration_size}")
        print(f"   (This reduces overhead by grouping small tasks)")
        
        # Prepare arguments for workers
        tasks = []
        for block_id in range(num_agglomerations):
            start_row = block_id * agglomeration_size
            end_row = min(start_row + agglomeration_size, n)
            tasks.append((block_id, start_row, end_row, A, B))
        
        # Execute agglomerated blocks
        with ProcessPoolExecutor(max_workers=num_agglomerations) as executor:
            results = executor.map(_compute_agglomerated_block_worker, tasks)
            
            for start_row, result in results:
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Agglomeration completed")
        return C
    
    # ==================== AMDAHL'S LAW ====================
    
    @staticmethod
    def demonstrate_amdahl_law(total_time: float = 10.0, 
                              parallelizable_fraction: float = 0.8):
        """
        Demonstrates Amdahl's Law
        
        Concept: Similar to AmdahlExample.java
        - Theoretical limit of speedup based on parallelizable fraction
        - Even with infinite processors, the serial part limits speedup
        
        Args:
            total_time: Total task time (seconds)
            parallelizable_fraction: Fraction that can be parallelized (0-1)
        """
        print(f"\n📐 AMDAHL'S LAW:")
        print(f"   Total time: {total_time}s")
        print(f"   Parallelizable fraction: {parallelizable_fraction*100:.0f}%")
        print(f"   Serial fraction: {(1-parallelizable_fraction)*100:.0f}%")
        print(f"\n   {'Processors':<15} {'Estimated Time (s)':<20} {'Speedup':<10}")
        print("   " + "-"*45)
        
        sequential_time = (1 - parallelizable_fraction) * total_time
        
        for processors in [1, 2, 4, 8, 16, 32, 64]:
            parallel_time = (parallelizable_fraction * total_time) / processors
            estimated_time = sequential_time + parallel_time
            speedup = total_time / estimated_time
            
            print(f"   {processors:<15} {estimated_time:<20.2f} {speedup:<10.2f}x")
        
        # Theoretical maximum speedup (with infinite processors)
        max_speedup = 1 / (1 - parallelizable_fraction)
        print(f"\n   💡 Theoretical maximum speedup: {max_speedup:.2f}x")
        print(f"      (with infinite processors)")


# ==================== DEMONSTRATION ====================

def demonstrate_concepts():
    """Demonstrates all concepts with small matrices"""
    print("="*70)
    print("DEMONSTRATION OF ADVANCED PARALLELIZATION CONCEPTS")
    print("="*70)
    
    # Small matrix for demonstration
    size = 100
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    concepts = AdvancedParallelConcepts()
    
    # 1. PARTITIONING
    print("\n" + "="*70)
    start = time.time()
    C1 = concepts.partitioning_multiplication(A, B, num_partitions=4)
    t1 = time.time() - start
    print(f"⏱️ Time: {t1:.4f}s")
    
    # 2. MAPPING
    print("\n" + "="*70)
    start = time.time()
    C2 = concepts.mapping_multiplication(A, B, num_threads=4)
    t2 = time.time() - start
    print(f"⏱️ Time: {t2:.4f}s")
    
    # 3. AGGLOMERATION
    print("\n" + "="*70)
    start = time.time()
    C3 = concepts.agglomeration_multiplication(A, B, num_agglomerations=2)
    t3 = time.time() - start
    print(f"⏱️ Time: {t3:.4f}s")
    
    # 4. AMDAHL'S LAW
    print("\n" + "="*70)
    concepts.demonstrate_amdahl_law(total_time=10.0, parallelizable_fraction=0.8)
    
    # Verify that all give the same result
    print("\n" + "="*70)
    print("✓ RESULT VERIFICATION:")
    reference = np.dot(A, B)
    print(f"   Partitioning correct: {np.allclose(C1, reference)}")
    print(f"   Mapping correct: {np.allclose(C2, reference)}")
    print(f"   Agglomeration correct: {np.allclose(C3, reference)}")
    
    print("\n" + "="*70)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("="*70)
    print("""
1. PARTITIONING (Data partitioning)
   → Divides data into independent blocks
   → Each block is processed in parallel
   → Good for uniform load distribution

2. MAPPING (Task mapping)
   → Assigns specific tasks to specific threads/processes
   → Explicit control over what worker does what
   → Useful for specific optimizations

3. AGGLOMERATION (Task agglomeration)
   → Groups small tasks into larger tasks
   → Reduces overhead of task management
   → Balance between parallelism and efficiency

4. AMDAHL'S LAW
   → Theoretical limit of speedup
   → Serial part limits maximum speedup
   → Speedup_max = 1 / (serial_fraction)
    """)


if __name__ == "__main__":
    demonstrate_concepts()