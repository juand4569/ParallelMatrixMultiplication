"""
benchmark_runner.py
Script to execute benchmarks of matrix multiplication

Task 3: Parallel Matrix Multiplication
Big Data - University of Las Palmas de Gran Canaria
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
    """Runs and records benchmarks of matrix multiplication"""
    
    def __init__(self, num_cores: int = None):
        """
        Initialize the runner
        
        Args:
            num_cores: Number of cores to use (None = all)
        """
        self.multiplier = MatrixMultiplier(num_cores)
        self.num_cores = self.multiplier.num_cores
        self.results = []
        
        # System info
        mem = psutil.virtual_memory()
        
        print("="*70)
        print("BENCHMARK SYSTEM INFORMATION")
        print("="*70)
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Processor: {platform.processor()}")
        print(f"Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"Logical cores (threads): {psutil.cpu_count(logical=True)}")
        print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        print(f"RAM usage: {mem.percent}%")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
    
    def benchmark_method(self, method_name: str, method_func, 
                        A: np.ndarray, B: np.ndarray, **kwargs) -> dict:
        """
        Executes benchmark of a specific method
        
        Args:
            method_name: Descriptive name of the method
            method_func: Function to execute
            A, B: Input matrices
            **kwargs: Additional arguments for method_func
            
        Returns:
            Dictionary with performance metrics
        """
        process = psutil.Process(os.getpid())
        
        # Measure memory before
        mem_before = process.memory_info().rss / (1024**2)  # MB
        
        # Warmup (especially important for Numba)
        if 'numba' in method_name.lower() and A.shape[0] <= 100:
            _ = method_func(A[:10, :10], B[:10, :10], **kwargs)
        
        # Measure time
        start_time = time.perf_counter()
        
        try:
            result = method_func(A, B, **kwargs)
            end_time = time.perf_counter()
            
            # Measure memory after
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
        Executes complete benchmark suite
        
        Args:
            matrix_sizes: List of matrix sizes to test
            methods: List of methods to test (None = all)
            num_threads_list: List of thread configurations
            num_processes_list: List of process configurations
            
        Returns:
            DataFrame with all results
        """
        if methods is None:
            methods = ['basic', 'threading', 'multiprocessing', 'numpy']
            if NUMBA_AVAILABLE:
                methods.append('numba')
        
        if num_threads_list is None:
            num_threads_list = [2, 4, self.num_cores]
        
        if num_processes_list is None:
            num_processes_list = [2, 4, self.num_cores]
        
        # Filter invalid configurations
        num_threads_list = [n for n in num_threads_list if n <= self.num_cores]
        num_processes_list = [n for n in num_processes_list if n <= self.num_cores]
        
        for size in matrix_sizes:
            print(f"\n{'='*70}")
            print(f"MATRICES {size}x{size}")
            print(f"{'='*70}")
            
            # Generate random matrices
            np.random.seed(42)  # For reproducibility
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # 1. Basic method (only for small matrices)
            if 'basic' in methods and size <= 500:
                result = self.benchmark_method(
                    "Basic (sequential)",
                    self.multiplier.basic_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size, 'threads': 1, 'processes': 1})
            
            # 2. Threading with different configurations
            if 'threading' in methods:
                for num_threads in num_threads_list:
                    result = self.benchmark_method(
                        f"Threading ({num_threads} threads)",
                        self.multiplier.threading_multiplication,
                        A, B,
                        num_threads=num_threads
                    )
                    self.results.append({**result, 'size': size, 'threads': num_threads})
            
            # 3. Multiprocessing with different configurations
            if 'multiprocessing' in methods:
                for num_processes in num_processes_list:
                    result = self.benchmark_method(
                        f"Multiprocessing ({num_processes} processes)",
                        self.multiplier.multiprocessing_multiplication,
                        A, B,
                        num_processes=num_processes
                    )
                    self.results.append({**result, 'size': size, 'processes': num_processes})
            
            # 4. NumPy (vectorized)
            if 'numpy' in methods:
                result = self.benchmark_method(
                    "NumPy (vectorized + parallel)",
                    self.multiplier.numpy_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
            
            # 5. Numba parallel (if available)
            if 'numba' in methods and NUMBA_AVAILABLE:
                result = self.benchmark_method(
                    "Numba parallel (JIT + prange)",
                    self.multiplier.numba_parallel_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
                
                # 6. Numba vectorized (if available)
                result = self.benchmark_method(
                    "Numba vectorized (JIT + dot)",
                    self.multiplier.numba_vectorized_multiplication,
                    A, B
                )
                self.results.append({**result, 'size': size})
        
        return pd.DataFrame(self.results)
    
    def save_results(self, df: pd.DataFrame, filename: str = 'benchmark_results.csv'):
        """Saves results to CSV"""
        df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to: {filename}")
    
    def print_summary(self, df: pd.DataFrame):
        """Prints summary of results"""
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        # Filter only successful results
        df_success = df[df['success'] == True].copy()
        
        if df_success.empty:
            print("No successful results to display")
            return
        
        # Best method by size
        print("\nBEST METHOD BY MATRIX SIZE:")
        print("-"*70)
        for size in sorted(df_success['size'].unique()):
            size_data = df_success[df_success['size'] == size]
            best = size_data.loc[size_data['execution_time'].idxmin()]
            print(f"  {size:4d}x{size:<4d}: {best['method']:35s} ({best['execution_time']:.4f}s)")
        
        # Average time
        print("\nAVERAGE TIME BY METHOD:")
        print("-"*70)
        avg_times = df_success.groupby('method')['execution_time'].mean().sort_values()
        for method, time in avg_times.items():
            print(f"  {method:45s}: {time:.4f}s")
        
        # Fastest method
        fastest = avg_times.index[0]
        fastest_time = avg_times.values[0]
        print(f"\n🏆 Fastest method: {fastest} ({fastest_time:.4f}s average)")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("BENCHMARK OF PARALLEL MATRIX MULTIPLICATION")
    print("Task 3 - Big Data - ULPGC")
    print("="*70 + "\n")
    
    # Create runner
    runner = BenchmarkRunner()
    
    # Benchmark configuration
    matrix_sizes = [100, 200, 500, 1000]
    
    # Adjust these configurations according to your system
    num_threads_list = [2, 4, runner.num_cores]
    num_processes_list = [2, 4, runner.num_cores]
    
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Thread configurations: {num_threads_list}")
    print(f"Process configurations: {num_processes_list}\n")
    
    # Execute benchmarks
    df_results = runner.run_benchmark_suite(
        matrix_sizes=matrix_sizes,
        num_threads_list=num_threads_list,
        num_processes_list=num_processes_list
    )
    
    filename = f'benchmark_results.csv'
    runner.save_results(df_results, filename)
    
    # Display summary
    runner.print_summary(df_results)
    
    print("\n✓ Benchmark completed!")
    print(f"\nYou can now analyze the results with:")
    print(f"  - analysis_notebook.ipynb (Jupyter Notebook)")
    print(f"  - Or directly with: python analyze_results.py {filename}")


if __name__ == "__main__":
    main()