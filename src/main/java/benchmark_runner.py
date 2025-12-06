"""
benchmark_runner.py
Script para ejecutar benchmarks de multiplicación de matrices

Tarea 3: Multiplicación de Matrices Paralela
Big Data - Universidad de Las Palmas de Gran Canaria
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
from datetime import datetime
import platform
from matrix_operations import MatrixMultiplier, NUMBA_AVAILABLE


class BenchmarkRunner:
    """Ejecuta y registra benchmarks de multiplicación de matrices"""
    
    def __init__(self, num_cores: int = None):
        """
        Inicializa el runner
        
        Args:
            num_cores: Número de núcleos a usar (None = todos)
        """
        self.multiplier = MatrixMultiplier(num_cores)
        self.num_cores = self.multiplier.num_cores
        self.results = []
        
        # Info del sistema (inspirado en printSystemInfo de Java)
        mem = psutil.virtual_memory()
        
        print("="*70)
        print("INFORMACIÓN DEL SISTEMA")
        print("="*70)
        print(f"Sistema Operativo: {platform.system()} {platform.release()}")
        print(f"Arquitectura: {platform.machine()}")
        print(f"Procesador: {platform.processor()}")
        print(f"Núcleos físicos: {psutil.cpu_count(logical=False)}")
        print(f"Núcleos lógicos (threads): {psutil.cpu_count(logical=True)}")
        print(f"RAM total: {mem.total / (1024**3):.2f} GB")
        print(f"RAM disponible: {mem.available / (1024**3):.2f} GB")
        print(f"RAM usada: {mem.percent}%")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
    
    def benchmark_method(self, method_name: str, method_func, 
                        A: np.ndarray, B: np.ndarray, **kwargs) -> dict:
        """
        Ejecuta benchmark de un método específico
        
        Args:
            method_name: Nombre descriptivo del método
            method_func: Función a ejecutar
            A, B: Matrices de entrada
            **kwargs: Argumentos adicionales para method_func
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        process = psutil.Process(os.getpid())
        
        # Medir memoria antes
        mem_before = process.memory_info().rss / (1024**2)  # MB
        
        # Warmup (especialmente importante para Numba)
        if 'numba' in method_name.lower() and A.shape[0] <= 100:
            _ = method_func(A[:10, :10], B[:10, :10], **kwargs)
        
        # Medir tiempo
        start_time = time.perf_counter()
        
        try:
            result = method_func(A, B, **kwargs)
            end_time = time.perf_counter()
            
            # Medir memoria después
            mem_after = process.memory_info().rss / (1024**2)  # MB
            
            execution_time = end_time - start_time
            memory_used = max(0, mem_after - mem_before)
            
            print(f"    ✓ {method_name:45s}: {execution_time:8.4f}s | Mem: {memory_used:6.1f}MB")
            
            return {
                'method': method_name,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'success': True
            }
            
        except Exception as e:
            print(f"    ✗ {method_name:45s}: ERROR - {str(e)}")
            return {
                'method': method_name,
                'execution_time': None,
                'memory_used': None,
                'success': False,
                'error': str(e)
            }
    
    def run_benchmark_suite(self, matrix_sizes: list, 
                           methods: list = None,
                           num_threads_list: list = None,
                           num_processes_list: list = None) -> pd.DataFrame:
        """
        Ejecuta suite completa de benchmarks
        
        Args:
            matrix_sizes: Lista de tamaños de matriz a probar
            methods: Lista de métodos a probar (None = todos)
            num_threads_list: Lista de configuraciones de threads
            num_processes_list: Lista de configuraciones de procesos
            
        Returns:
            DataFrame con todos los resultados
        """
        if methods is None:
            methods = ['basic', 'threading', 'multiprocessing', 'numpy']
            if NUMBA_AVAILABLE:
                methods.append('numba')
        
        if num_threads_list is None:
            num_threads_list = [2, 4, self.num_cores]
        
        if num_processes_list is None:
            num_processes_list = [2, 4, self.num_cores]
        
        # Filtrar configuraciones inválidas
        num_threads_list = [n for n in num_threads_list if n <= self.num_cores]
        num_processes_list = [n for n in num_processes_list if n <= self.num_cores]
        
        for size in matrix_sizes:
            print(f"\n{'='*70}")
            print(f"MATRICES {size}x{size}")
            print(f"{'='*70}")
            
            # Generar matrices aleatorias
            np.random.seed(42)  # Para reproducibilidad
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # 1. Método básico (solo para matrices pequeñas)
            if 'basic' in methods and size <= 500:
                result = self.benchmark_method(
                    "Básico (secuencial)",
                    self.multiplier.basic_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size, 'threads': 1, 'processes': 1})
            
            # 2. Threading con diferentes configuraciones
            if 'threading' in methods:
                for num_threads in num_threads_list:
                    result = self.benchmark_method(
                        f"Threading ({num_threads} threads)",
                        self.multiplier.threading_multiplication,
                        A, B,
                        num_threads=num_threads
                    )
                    self.results.append({**result, 'size': size, 'threads': num_threads})
            
            # 3. Multiprocessing con diferentes configuraciones
            if 'multiprocessing' in methods:
                for num_processes in num_processes_list:
                    result = self.benchmark_method(
                        f"Multiprocessing ({num_processes} procesos)",
                        self.multiplier.multiprocessing_multiplication,
                        A, B,
                        num_processes=num_processes
                    )
                    self.results.append({**result, 'size': size, 'processes': num_processes})
            
            # 4. NumPy (vectorizado)
            if 'numpy' in methods:
                result = self.benchmark_method(
                    "NumPy (vectorizado + paralelo)",
                    self.multiplier.numpy_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
            
            # 5. Numba paralelo (si está disponible)
            if 'numba' in methods and NUMBA_AVAILABLE:
                result = self.benchmark_method(
                    "Numba paralelo (JIT + prange)",
                    self.multiplier.numba_parallel_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
                
                # 6. Numba vectorizado (si está disponible)
                result = self.benchmark_method(
                    "Numba vectorizado (JIT + dot)",
                    self.multiplier.numba_vectorized_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
        
        return pd.DataFrame(self.results)
    
    def save_results(self, df: pd.DataFrame, filename: str = 'benchmark_results.csv'):
        """Guarda resultados en CSV"""
        df.to_csv(filename, index=False)
        print(f"\n✓ Resultados guardados en: {filename}")
    
    def print_summary(self, df: pd.DataFrame):
        """Imprime resumen de resultados"""
        print("\n" + "="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        
        # Filtrar solo resultados exitosos
        df_success = df[df['success'] == True].copy()
        
        if df_success.empty:
            print("No hay resultados exitosos para mostrar")
            return
        
        # Mejor método por tamaño
        print("\nMEJOR MÉTODO POR TAMAÑO DE MATRIZ:")
        print("-"*70)
        for size in sorted(df_success['size'].unique()):
            size_data = df_success[df_success['size'] == size]
            best = size_data.loc[size_data['execution_time'].idxmin()]
            print(f"  {size:4d}x{size:<4d}: {best['method']:35s} ({best['execution_time']:.4f}s)")
        
        # Promedio general
        print("\nTIEMPO PROMEDIO POR MÉTODO:")
        print("-"*70)
        avg_times = df_success.groupby('method')['execution_time'].mean().sort_values()
        for method, time in avg_times.items():
            print(f"  {method:45s}: {time:.4f}s")
        
        # Método más rápido
        fastest = avg_times.index[0]
        fastest_time = avg_times.values[0]
        print(f"\n🏆 Método más rápido: {fastest} ({fastest_time:.4f}s promedio)")


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("BENCHMARK DE MULTIPLICACIÓN DE MATRICES PARALELA")
    print("Tarea 3 - Big Data - ULPGC")
    print("="*70 + "\n")
    
    # Crear runner
    runner = BenchmarkRunner()
    
    # Configuración del benchmark
    matrix_sizes = [100, 200, 500, 1000]
    
    # Ajusta estas configuraciones según tu sistema
    num_threads_list = [2, 4, runner.num_cores]
    num_processes_list = [2, 4, runner.num_cores]
    
    print(f"Tamaños de matriz: {matrix_sizes}")
    print(f"Configuraciones de threads: {num_threads_list}")
    print(f"Configuraciones de procesos: {num_processes_list}")
    
    input("\nPresiona ENTER para comenzar el benchmark...")
    
    # Ejecutar benchmarks
    df_results = runner.run_benchmark_suite(
        matrix_sizes=matrix_sizes,
        num_threads_list=num_threads_list,
        num_processes_list=num_processes_list
    )
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'benchmark_results_{timestamp}.csv'
    runner.save_results(df_results, filename)
    
    # Mostrar resumen
    runner.print_summary(df_results)
    
    print("\n✓ Benchmark completado!")
    print(f"\nAhora puedes analizar los resultados con:")
    print(f"  - analysis_notebook.ipynb (Jupyter Notebook)")
    print(f"  - O directamente con: python analyze_results.py {filename}")


if __name__ == "__main__":
    main()