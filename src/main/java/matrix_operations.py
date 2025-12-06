"""
matrix_operations.py
Implementations of matrix multiplication with different approaches

Task 3: Parallel Matrix Multiplication
Big Data - University of Las Palmas de Gran Canaria
"""

import numpy as np
import threading
import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple
import warnings

# Try to import numba (optional)
try:
    from numba import jit, prange
    from numba.core.errors import NumbaPerformanceWarning
    NUMBA_AVAILABLE = True
    # Suppress Numba performance warnings
    warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not available. Install with: pip install numba")


class MatrixMultiplier:
    """Class with different matrix multiplication implementations"""
    
    def __init__(self, num_cores: int = None):
        """
        Initialize the multiplier
        
        Args:
            num_cores: Number of cores to use (None = all)
        """
        self.num_cores = num_cores or mp.cpu_count()
    
    def basic_multiplication(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Basic O(n³) multiplication - Sequential
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            
        Returns:
            Result matrix nxp
        """
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Incompatible dimensions: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    def threading_multiplication(self, A: np.ndarray, B: np.ndarray, 
                                num_threads: int = None) -> np.ndarray:
        """
        Parallel multiplication with threading
        Divides rows among multiple threads
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            num_threads: Number of threads (default: all cores)
            
        Returns:
            Result matrix nxp
        """
        if num_threads is None:
            num_threads = self.num_cores
            
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Incompatible dimensions: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        
        def compute_rows(start_row: int, end_row: int):
            """Computes a block of rows"""
            for i in range(start_row, end_row):
                for j in range(p):
                    for k in range(m):
                        C[i, j] += A[i, k] * B[k, j]
        
        # Create and launch threads
        threads = []
        rows_per_thread = n // num_threads
        
        for t in range(num_threads):
            start_row = t * rows_per_thread
            end_row = (t + 1) * rows_per_thread if t < num_threads - 1 else n
            
            thread = threading.Thread(target=compute_rows, args=(start_row, end_row))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return C
    
    @staticmethod
    def _worker_function(args: Tuple) -> Tuple[int, np.ndarray]:
        """
        Worker function for multiprocessing
        Computes a block of rows of the result matrix
        """
        start_row, end_row, A, B = args
        n, m = A.shape
        m2, p = B.shape
        
        result = np.zeros((end_row - start_row, p))
        for i in range(end_row - start_row):
            for j in range(p):
                for k in range(m):
                    result[i, j] += A[start_row + i, k] * B[k, j]
        
        return (start_row, result)
    
    def multiprocessing_multiplication(self, A: np.ndarray, B: np.ndarray,
                                      num_processes: int = None) -> np.ndarray:
        """
        Parallel multiplication with multiprocessing
        More efficient than threading (avoids the GIL)
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            num_processes: Number of processes (default: all cores)
            
        Returns:
            Result matrix nxp
        """
        if num_processes is None:
            num_processes = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Incompatible dimensions: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        
        # Divide work among processes
        rows_per_process = n // num_processes
        tasks = []
        
        for proc in range(num_processes):
            start_row = proc * rows_per_process
            end_row = (proc + 1) * rows_per_process if proc < num_processes - 1 else n
            tasks.append((start_row, end_row, A, B))
        
        # Execute in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._worker_function, tasks)
        
        # Combine results
        for start_row, result in results:
            end_row = start_row + result.shape[0]
            C[start_row:end_row, :] = result
        
        return C
    
    def numpy_multiplication(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplication with NumPy (vectorized and optimized)
        Uses BLAS/LAPACK internally (parallel and vectorized)
        
        Args:
            A: Matrix nxm
            B: Matrix mxp
            
        Returns:
            Result matrix nxp
        """
        return np.dot(A, B)
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def numba_parallel_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """
            Multiplication with parallel Numba
            JIT compilation + automatic parallelization
            Similar to the teacher's VectorizedMatrixMultiplication.py example
            
            Args:
                A: Matrix nxm
                B: Matrix mxp
                
            Returns:
                Result matrix nxp
            """
            n, m = A.shape
            m2, p = B.shape
            C = np.zeros((n, p))
            
            for i in prange(n):  # prange = parallel range
                for j in range(p):
                    for k in range(m):
                        C[i, j] += A[i, k] * B[k, j]
            return C
        
        @staticmethod
        @jit(nopython=True)
        def numba_vectorized_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """
            Multiplication with Numba using vectorized dot product
            Similar to the teacher's example but without explicit parallelization
            
            Args:
                A: Matrix nxm
                B: Matrix mxp
                
            Returns:
                Result matrix nxp
            """
            n, p = A.shape[0], B.shape[1]
            C = np.zeros((n, p))
            
            for i in range(n):
                for j in range(p):
                    # Producto punto vectorizado (como en el ejemplo de clase)
                    C[i, j] = np.dot(A[i, :], B[:, j])
            return C


def get_available_methods():
    """Returns list of available methods"""
    methods = [
        'basic',
        'threading', 
        'multiprocessing',
        'numpy'
    ]
    
    if NUMBA_AVAILABLE:
        methods.append('numba')
    
    return methods


def verify_implementation():
    """Verifies that all implementations produce the same result"""
    print("Verifying implementations...")
    
    # Small matrices for verification
    A = np.random.rand(50, 50)
    B = np.random.rand(50, 50)
    
    multiplier = MatrixMultiplier()
    
    # Reference result (NumPy)
    reference = multiplier.numpy_multiplication(A, B)
    
    # Verify each method
    methods = {
        'Basic': lambda: multiplier.basic_multiplication(A, B),
        'Threading': lambda: multiplier.threading_multiplication(A, B, 2),
        'Multiprocessing': lambda: multiplier.multiprocessing_multiplication(A, B, 2),
    }
    
    if NUMBA_AVAILABLE:
        methods['Numba'] = lambda: multiplier.numba_parallel_multiplication(A, B)
    
    all_correct = True
    for name, method in methods.items():
        result = method()
        if np.allclose(result, reference, rtol=1e-5):
            print(f"  ✓ {name}: Correct")
        else:
            print(f"  ✗ {name}: ERROR - Incorrect result")
            all_correct = False
    
    return all_correct


if __name__ == "__main__":
    # Verify implementations
    if verify_implementation():
        print("\n✓ All implementations are correct")
    else:
        print("\n✗ There are errors in the implementations")