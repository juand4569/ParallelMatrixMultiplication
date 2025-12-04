"""
matrix_operations.py
Implementaciones de multiplicación de matrices con diferentes enfoques

Tarea 3: Multiplicación de Matrices Paralela
Big Data - Universidad de Las Palmas de Gran Canaria
"""

import numpy as np
import threading
import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple

# Intentar importar numba (opcional)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba no disponible. Instalar con: pip install numba")


class MatrixMultiplier:
    """Clase con diferentes implementaciones de multiplicación de matrices"""
    
    def __init__(self, num_cores: int = None):
        """
        Inicializa el multiplicador
        
        Args:
            num_cores: Número de núcleos a usar (None = todos)
        """
        self.num_cores = num_cores or mp.cpu_count()
    
    def basic_multiplication(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplicación básica O(n³) - Secuencial
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            
        Returns:
            Matriz resultado nxp
        """
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Dimensiones incompatibles: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    def threading_multiplication(self, A: np.ndarray, B: np.ndarray, 
                                num_threads: int = None) -> np.ndarray:
        """
        Multiplicación paralela con threading
        Divide las filas entre múltiples threads
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            num_threads: Número de threads (default: todos los cores)
            
        Returns:
            Matriz resultado nxp
        """
        if num_threads is None:
            num_threads = self.num_cores
            
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Dimensiones incompatibles: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        
        def compute_rows(start_row: int, end_row: int):
            """Calcula un bloque de filas"""
            for i in range(start_row, end_row):
                for j in range(p):
                    for k in range(m):
                        C[i, j] += A[i, k] * B[k, j]
        
        # Crear y lanzar threads
        threads = []
        rows_per_thread = n // num_threads
        
        for t in range(num_threads):
            start_row = t * rows_per_thread
            end_row = (t + 1) * rows_per_thread if t < num_threads - 1 else n
            
            thread = threading.Thread(target=compute_rows, args=(start_row, end_row))
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen todos
        for thread in threads:
            thread.join()
        
        return C
    
    @staticmethod
    def _worker_function(args: Tuple) -> Tuple[int, np.ndarray]:
        """
        Función worker para multiprocessing
        Calcula un bloque de filas de la matriz resultado
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
        Multiplicación paralela con multiprocessing
        Más eficiente que threading (evita el GIL)
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            num_processes: Número de procesos (default: todos los cores)
            
        Returns:
            Matriz resultado nxp
        """
        if num_processes is None:
            num_processes = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, f"Dimensiones incompatibles: {A.shape} x {B.shape}"
        
        C = np.zeros((n, p))
        
        # Dividir trabajo entre procesos
        rows_per_process = n // num_processes
        tasks = []
        
        for proc in range(num_processes):
            start_row = proc * rows_per_process
            end_row = (proc + 1) * rows_per_process if proc < num_processes - 1 else n
            tasks.append((start_row, end_row, A, B))
        
        # Ejecutar en paralelo
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._worker_function, tasks)
        
        # Combinar resultados
        for start_row, result in results:
            end_row = start_row + result.shape[0]
            C[start_row:end_row, :] = result
        
        return C
    
    def numpy_multiplication(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplicación con NumPy (vectorizada y optimizada)
        Usa BLAS/LAPACK internamente (paralelo y vectorizado)
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            
        Returns:
            Matriz resultado nxp
        """
        return np.dot(A, B)
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def numba_parallel_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """
            Multiplicación con Numba paralelo
            JIT compilation + paralelización automática
            
            Args:
                A: Matriz nxm
                B: Matriz mxp
                
            Returns:
                Matriz resultado nxp
            """
            n, m = A.shape
            m2, p = B.shape
            C = np.zeros((n, p))
            
            for i in prange(n):  # prange = parallel range
                for j in range(p):
                    for k in range(m):
                        C[i, j] += A[i, k] * B[k, j]
            return C


def get_available_methods():
    """Retorna lista de métodos disponibles"""
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
    """Verifica que todas las implementaciones dan el mismo resultado"""
    print("Verificando implementaciones...")
    
    # Matrices pequeñas para verificación
    A = np.random.rand(50, 50)
    B = np.random.rand(50, 50)
    
    multiplier = MatrixMultiplier()
    
    # Resultado de referencia (NumPy)
    reference = multiplier.numpy_multiplication(A, B)
    
    # Verificar cada método
    methods = {
        'Básico': lambda: multiplier.basic_multiplication(A, B),
        'Threading': lambda: multiplier.threading_multiplication(A, B, 2),
        'Multiprocessing': lambda: multiplier.multiprocessing_multiplication(A, B, 2),
    }
    
    if NUMBA_AVAILABLE:
        methods['Numba'] = lambda: multiplier.numba_parallel_multiplication(A, B)
    
    all_correct = True
    for name, method in methods.items():
        result = method()
        if np.allclose(result, reference, rtol=1e-5):
            print(f"  ✓ {name}: Correcto")
        else:
            print(f"  ✗ {name}: ERROR - Resultado incorrecto")
            all_correct = False
    
    return all_correct


if __name__ == "__main__":
    # Verificar implementaciones
    if verify_implementation():
        print("\n✓ Todas las implementaciones son correctas")
    else:
        print("\n✗ Hay errores en las implementaciones")