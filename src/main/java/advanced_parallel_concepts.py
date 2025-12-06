"""
advanced_parallel_concepts.py
Implementación de conceptos avanzados de paralelización
Basado en los ejemplos de la profesora (Partitioning, Mapping, Agglomeration)

Tarea 3: Multiplicación de Matrices Paralela
Big Data - Universidad de Las Palmas de Gran Canaria
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple, List


# ==================== FUNCIONES AUXILIARES (FUERA DE LA CLASE) ====================
# Necesarias para Windows multiprocessing

def _compute_partition_worker(args):
    """Worker para partitioning - debe estar a nivel módulo para Windows"""
    partition_id, start_row, end_row, A, B = args
    m, p = A.shape[1], B.shape[1]
    
    print(f"   Partición {partition_id}: filas {start_row} a {end_row-1}")
    
    result = np.zeros((end_row - start_row, p))
    for i in range(end_row - start_row):
        for j in range(p):
            for k in range(m):
                result[i, j] += A[start_row + i, k] * B[k, j]
    
    return (start_row, result)


def _compute_agglomerated_block_worker(args):
    """Worker para agglomeration - debe estar a nivel módulo para Windows"""
    block_id, start_row, end_row, A, B = args
    m, p = A.shape[1], B.shape[1]
    
    print(f"   Bloque-{block_id}: procesa {end_row - start_row} filas")
    
    result = np.zeros((end_row - start_row, p))
    for i in range(end_row - start_row):
        for j in range(p):
            for k in range(m):
                result[i, j] += A[start_row + i, k] * B[k, j]
    
    return (start_row, result)


class AdvancedParallelConcepts:
    """
    Implementaciones que demuestran conceptos avanzados de paralelización:
    - Partitioning (Particionamiento)
    - Mapping (Mapeo)
    - Agglomeration (Aglomeración)
    
    Basado en los ejemplos Java de la profesora pero adaptados a Python
    """
    
    def __init__(self, num_cores: int = None):
        self.num_cores = num_cores or mp.cpu_count()
    
    # ==================== PARTITIONING ====================
    
    def partitioning_multiplication(self, A: np.ndarray, B: np.ndarray, 
                                   num_partitions: int = None) -> np.ndarray:
        """
        PARTITIONING: Divide los datos en bloques independientes
        
        Concepto: Similar a PartitioningExample.java
        - Divide la matriz A en particiones horizontales
        - Cada partición calcula su resultado independientemente
        - Los resultados se combinan al final
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            num_partitions: Número de particiones (bloques)
        
        Returns:
            Matriz resultado nxp
        """
        if num_partitions is None:
            num_partitions = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Dimensiones incompatibles"
        
        C = np.zeros((n, p))
        
        # Calcular tamaño de cada partición
        partition_size = int(np.ceil(n / num_partitions))
        
        print(f"\n📦 PARTITIONING:")
        print(f"   Total filas: {n}")
        print(f"   Particiones: {num_partitions}")
        print(f"   Filas por partición: {partition_size}")
        
        # Preparar argumentos para workers
        tasks = []
        for part_id in range(num_partitions):
            start_row = part_id * partition_size
            end_row = min(start_row + partition_size, n)
            tasks.append((part_id, start_row, end_row, A, B))
        
        # Usar ProcessPoolExecutor para verdadero paralelismo
        with ProcessPoolExecutor(max_workers=num_partitions) as executor:
            results = executor.map(_compute_partition_worker, tasks)
            
            # Recolectar resultados
            for start_row, result in results:
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Partitioning completado")
        return C
    
    # ==================== MAPPING ====================
    
    def mapping_multiplication(self, A: np.ndarray, B: np.ndarray,
                              num_threads: int = None) -> np.ndarray:
        """
        MAPPING: Asigna tareas específicas a threads/procesos concretos
        
        Concepto: Similar a MappingExample.java
        - Cada thread tiene un ID único y un bloque de trabajo asignado
        - Se mapea explícitamente qué thread procesa qué datos
        - Útil cuando queremos control sobre la asignación de tareas
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            num_threads: Número de threads a mapear
        
        Returns:
            Matriz resultado nxp
        """
        if num_threads is None:
            num_threads = self.num_cores
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Dimensiones incompatibles"
        
        C = np.zeros((n, p))
        block_size = int(np.ceil(n / num_threads))
        
        print(f"\n🗺️ MAPPING:")
        print(f"   Threads disponibles: {num_threads}")
        print(f"   Bloque por thread: {block_size} filas")
        
        def mapped_task(thread_id: int) -> Tuple[int, int, np.ndarray]:
            """Tarea mapeada a un thread específico"""
            start_row = thread_id * block_size
            end_row = min(start_row + block_size, n)
            
            print(f"   Thread-{thread_id} → filas [{start_row}:{end_row-1}]")
            
            result = np.zeros((end_row - start_row, p))
            for i in range(end_row - start_row):
                for j in range(p):
                    for k in range(m):
                        result[i, j] += A[start_row + i, k] * B[k, j]
            
            return (thread_id, start_row, result)
        
        # Mapear explícitamente tareas a threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mapped_task, tid) for tid in range(num_threads)]
            
            for future in as_completed(futures):
                thread_id, start_row, result = future.result()
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Mapping completado")
        return C
    
    # ==================== AGGLOMERATION ====================
    
    def agglomeration_multiplication(self, A: np.ndarray, B: np.ndarray,
                                    num_agglomerations: int = None) -> np.ndarray:
        """
        AGGLOMERATION: Agrupa tareas pequeñas en tareas más grandes
        
        Concepto: Similar a AgglomerationExample.java
        - En vez de crear una tarea por fila, agrupamos múltiples filas
        - Reduce el overhead de creación/gestión de tareas
        - Balance entre paralelismo y overhead
        
        Args:
            A: Matriz nxm
            B: Matriz mxp
            num_agglomerations: Número de grupos grandes (< num_filas)
        
        Returns:
            Matriz resultado nxp
        """
        if num_agglomerations is None:
            # Por defecto: agglomeration = cores/2 (menos tareas, más trabajo cada una)
            num_agglomerations = max(1, self.num_cores // 2)
        
        n, m = A.shape
        m2, p = B.shape
        assert m == m2, "Dimensiones incompatibles"
        
        C = np.zeros((n, p))
        
        # Tamaño grande de cada aglomeración
        agglomeration_size = int(np.ceil(n / num_agglomerations))
        
        print(f"\n🔗 AGGLOMERATION:")
        print(f"   Total filas: {n}")
        print(f"   Aglomeraciones: {num_agglomerations}")
        print(f"   Filas por aglomeración: {agglomeration_size}")
        print(f"   (Esto reduce el overhead al agrupar tareas pequeñas)")
        
        # Preparar argumentos para workers
        tasks = []
        for block_id in range(num_agglomerations):
            start_row = block_id * agglomeration_size
            end_row = min(start_row + agglomeration_size, n)
            tasks.append((block_id, start_row, end_row, A, B))
        
        # Ejecutar bloques aglomerados
        with ProcessPoolExecutor(max_workers=num_agglomerations) as executor:
            results = executor.map(_compute_agglomerated_block_worker, tasks)
            
            for start_row, result in results:
                end_row = start_row + result.shape[0]
                C[start_row:end_row, :] = result
        
        print(f"   ✓ Agglomeration completado")
        return C
    
    # ==================== LEY DE AMDAHL ====================
    
    @staticmethod
    def demonstrate_amdahl_law(total_time: float = 10.0, 
                              parallelizable_fraction: float = 0.8):
        """
        Demuestra la Ley de Amdahl
        
        Concepto: Similar a AmdahlExample.java
        - Límite teórico del speedup basado en la fracción paralelizable
        - Incluso con infinitos procesadores, la parte serial limita el speedup
        
        Args:
            total_time: Tiempo total de la tarea (segundos)
            parallelizable_fraction: Fracción que se puede paralelizar (0-1)
        """
        print(f"\n📐 LEY DE AMDAHL:")
        print(f"   Tiempo total: {total_time}s")
        print(f"   Fracción paralelizable: {parallelizable_fraction*100:.0f}%")
        print(f"   Fracción serial: {(1-parallelizable_fraction)*100:.0f}%")
        print(f"\n   {'Procesadores':<15} {'Tiempo Estimado (s)':<20} {'Speedup':<10}")
        print("   " + "-"*45)
        
        sequential_time = (1 - parallelizable_fraction) * total_time
        
        for processors in [1, 2, 4, 8, 16, 32, 64]:
            parallel_time = (parallelizable_fraction * total_time) / processors
            estimated_time = sequential_time + parallel_time
            speedup = total_time / estimated_time
            
            print(f"   {processors:<15} {estimated_time:<20.2f} {speedup:<10.2f}x")
        
        # Speedup máximo teórico (con infinitos procesadores)
        max_speedup = 1 / (1 - parallelizable_fraction)
        print(f"\n   💡 Speedup máximo teórico: {max_speedup:.2f}x")
        print(f"      (con infinitos procesadores)")


# ==================== DEMOSTRACIÓN ====================

def demonstrate_concepts():
    """Demuestra todos los conceptos con matrices pequeñas"""
    print("="*70)
    print("DEMOSTRACIÓN DE CONCEPTOS AVANZADOS DE PARALELIZACIÓN")
    print("="*70)
    
    # Matriz pequeña para demostración
    size = 100
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    concepts = AdvancedParallelConcepts()
    
    # 1. PARTITIONING
    print("\n" + "="*70)
    start = time.time()
    C1 = concepts.partitioning_multiplication(A, B, num_partitions=4)
    t1 = time.time() - start
    print(f"⏱️ Tiempo: {t1:.4f}s")
    
    # 2. MAPPING
    print("\n" + "="*70)
    start = time.time()
    C2 = concepts.mapping_multiplication(A, B, num_threads=4)
    t2 = time.time() - start
    print(f"⏱️ Tiempo: {t2:.4f}s")
    
    # 3. AGGLOMERATION
    print("\n" + "="*70)
    start = time.time()
    C3 = concepts.agglomeration_multiplication(A, B, num_agglomerations=2)
    t3 = time.time() - start
    print(f"⏱️ Tiempo: {t3:.4f}s")
    
    # 4. LEY DE AMDAHL
    print("\n" + "="*70)
    concepts.demonstrate_amdahl_law(total_time=10.0, parallelizable_fraction=0.8)
    
    # Verificar que todos dan el mismo resultado
    print("\n" + "="*70)
    print("✓ VERIFICACIÓN DE RESULTADOS:")
    reference = np.dot(A, B)
    print(f"   Partitioning correcto: {np.allclose(C1, reference)}")
    print(f"   Mapping correcto: {np.allclose(C2, reference)}")
    print(f"   Agglomeration correcto: {np.allclose(C3, reference)}")
    
    print("\n" + "="*70)
    print("CONCEPTOS CLAVE DEMOSTRADOS:")
    print("="*70)
    print("""
1. PARTITIONING (Particionamiento)
   → Divide los datos en bloques independientes
   → Cada bloque se procesa en paralelo
   → Buenos para distribución de carga uniforme

2. MAPPING (Mapeo)
   → Asigna tareas específicas a threads/procesos concretos
   → Control explícito sobre qué worker hace qué
   → Útil para optimizaciones específicas

3. AGGLOMERATION (Aglomeración)
   → Agrupa tareas pequeñas en tareas más grandes
   → Reduce overhead de gestión de tareas
   → Balance entre paralelismo y eficiencia

4. LEY DE AMDAHL
   → Límite teórico del speedup
   → La parte serial limita el speedup máximo
   → Speedup_max = 1 / (fracción_serial)
    """)


if __name__ == "__main__":
    demonstrate_concepts()