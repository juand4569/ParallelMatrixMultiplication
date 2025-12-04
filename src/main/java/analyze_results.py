"""
analyze_results.py
Script para analizar y visualizar resultados de benchmarks

Tarea 3: Multiplicación de Matrices Paralela
Big Data - Universidad de Las Palmas de Gran Canaria
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from pathlib import Path


class ResultsAnalyzer:
    """Analiza y visualiza resultados de benchmarks"""
    
    def __init__(self, csv_file: str):
        """
        Inicializa el analizador
        
        Args:
            csv_file: Ruta al archivo CSV con resultados
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        
        # Filtrar solo resultados exitosos
        self.df = self.df[self.df['success'] == True].copy()
        
        print(f"✓ Cargados {len(self.df)} resultados desde {csv_file}")
        print(f"  Tamaños de matriz: {sorted(self.df['size'].unique())}")
        print(f"  Métodos: {self.df['method'].unique().tolist()}")
    
    def calculate_speedup(self) -> pd.DataFrame:
        """
        Calcula speedup respecto al método básico
        Si no hay básico, usa el más lento como referencia
        """
        df = self.df.copy()
        df['speedup'] = np.nan
        
        for size in df['size'].unique():
            size_mask = df['size'] == size
            size_data = df[size_mask]
            
            # Buscar tiempo baseline (método básico)
            baseline = size_data[size_data['method'] == 'Básico (secuencial)']
            
            if not baseline.empty:
                baseline_time = baseline['execution_time'].values[0]
            else:
                # Si no hay básico, usar el más lento
                baseline_time = size_data['execution_time'].max()
            
            # Calcular speedup para este tamaño
            df.loc[size_mask, 'speedup'] = baseline_time / df.loc[size_mask, 'execution_time']
        
        return df
    
    def calculate_efficiency(self) -> pd.DataFrame:
        """
        Calcula eficiencia = speedup / número de cores
        """
        df = self.calculate_speedup()
        df['efficiency'] = np.nan
        
        # Para threading
        threading_mask = df['method'].str.contains('Threading', na=False)
        if 'threads' in df.columns:
            df.loc[threading_mask, 'efficiency'] = (
                df.loc[threading_mask, 'speedup'] / df.loc[threading_mask, 'threads']
            )
        
        # Para multiprocessing
        mp_mask = df['method'].str.contains('Multiprocessing', na=False)
        if 'processes' in df.columns:
            df.loc[mp_mask, 'efficiency'] = (
                df.loc[mp_mask, 'speedup'] / df.loc[mp_mask, 'processes']
            )
        
        return df
    
    def plot_execution_time(self, save_path: str = 'plot_execution_time.png'):
        """Gráfico: Tiempo de ejecución vs tamaño de matriz"""
        df = self.df.copy()
        
        plt.figure(figsize=(12, 7))
        
        # Colores consistentes
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        color_map = dict(zip(methods, colors))
        
        for method in methods:
            method_data = df[df['method'] == method].sort_values('size')
            plt.plot(method_data['size'], method_data['execution_time'],
                    marker='o', linewidth=2, markersize=8,
                    label=method, color=color_map[method])
        
        plt.xlabel('Tamaño de Matriz (nxn)', fontsize=12)
        plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
        plt.title('Tiempo de Ejecución vs Tamaño de Matriz', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()
    
    def plot_speedup(self, save_path: str = 'plot_speedup.png'):
        """Gráfico: Speedup vs tamaño de matriz"""
        df = self.calculate_speedup()
        
        plt.figure(figsize=(12, 7))
        
        # Excluir método básico del gráfico de speedup
        df_plot = df[df['method'] != 'Básico (secuencial)'].copy()
        
        methods = df_plot['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        color_map = dict(zip(methods, colors))
        
        for method in methods:
            method_data = df_plot[df_plot['method'] == method].sort_values('size')
            plt.plot(method_data['size'], method_data['speedup'],
                    marker='s', linewidth=2, markersize=8,
                    label=method, color=color_map[method])
        
        # Línea de referencia (speedup = 1)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, 
                   linewidth=2, label='Baseline (speedup=1)')
        
        plt.xlabel('Tamaño de Matriz (nxn)', fontsize=12)
        plt.ylabel('Speedup (veces más rápido)', fontsize=12)
        plt.title('Speedup vs Tamaño de Matriz', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()
    
    def plot_efficiency(self, save_path: str = 'plot_efficiency.png'):
        """Gráfico: Eficiencia vs número de cores"""
        df = self.calculate_efficiency()
        
        # Filtrar solo métodos paralelos con eficiencia calculada
        df_plot = df[df['efficiency'].notna()].copy()
        
        if df_plot.empty:
            print("⚠️ No hay datos de eficiencia para graficar")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Threading
        threading_data = df_plot[df_plot['method'].str.contains('Threading', na=False)]
        if not threading_data.empty:
            for size in sorted(threading_data['size'].unique()):
                size_data = threading_data[threading_data['size'] == size].sort_values('threads')
                ax1.plot(size_data['threads'], size_data['efficiency'],
                        marker='o', linewidth=2, markersize=8,
                        label=f'{size}x{size}')
            
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.set_xlabel('Número de Threads', fontsize=11)
            ax1.set_ylabel('Eficiencia', fontsize=11)
            ax1.set_title('Eficiencia - Threading', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(1.2, threading_data['efficiency'].max() * 1.1))
        
        # 2. Multiprocessing
        mp_data = df_plot[df_plot['method'].str.contains('Multiprocessing', na=False)]
        if not mp_data.empty:
            for size in sorted(mp_data['size'].unique()):
                size_data = mp_data[mp_data['size'] == size].sort_values('processes')
                ax2.plot(size_data['processes'], size_data['efficiency'],
                        marker='s', linewidth=2, markersize=8,
                        label=f'{size}x{size}')
            
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax2.set_xlabel('Número de Procesos', fontsize=11)
            ax2.set_ylabel('Eficiencia', fontsize=11)
            ax2.set_title('Eficiencia - Multiprocessing', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(1.2, mp_data['efficiency'].max() * 1.1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()
    
    def plot_comparison_heatmap(self, save_path: str = 'plot_heatmap.png'):
        """Heatmap: Comparación de tiempos"""
        df = self.df.copy()
        
        # Crear matriz pivote
        pivot = df.pivot_table(
            values='execution_time',
            index='method',
            columns='size',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Tiempo (s)'})
        plt.title('Heatmap: Tiempo de Ejecución por Método y Tamaño', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Tamaño de Matriz (nxn)', fontsize=12)
        plt.ylabel('Método', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()
    
    def plot_memory_usage(self, save_path: str = 'plot_memory.png'):
        """Gráfico: Uso de memoria"""
        df = self.df.copy()
        
        plt.figure(figsize=(12, 7))
        
        methods = df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        color_map = dict(zip(methods, colors))
        
        for method in methods:
            method_data = df[df['method'] == method].sort_values('size')
            plt.plot(method_data['size'], method_data['memory_used'],
                    marker='o', linewidth=2, markersize=8,
                    label=method, color=color_map[method])
        
        plt.xlabel('Tamaño de Matriz (nxn)', fontsize=12)
        plt.ylabel('Memoria Usada (MB)', fontsize=12)
        plt.title('Uso de Memoria vs Tamaño de Matriz', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {save_path}")
        plt.close()
    
    def generate_all_plots(self, output_dir: str = 'plots'):
        """Genera todos los gráficos"""
        # Crear directorio si no existe
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerando gráficos en: {output_dir}/")
        print("-" * 60)
        
        self.plot_execution_time(f'{output_dir}/01_execution_time.png')
        self.plot_speedup(f'{output_dir}/02_speedup.png')
        self.plot_efficiency(f'{output_dir}/03_efficiency.png')
        self.plot_comparison_heatmap(f'{output_dir}/04_heatmap.png')
        self.plot_memory_usage(f'{output_dir}/05_memory_usage.png')
        
        print("-" * 60)
        print(f"✓ Todos los gráficos generados en: {output_dir}/")
    
    def generate_report(self, output_file: str = 'benchmark_report.txt'):
        """Genera reporte detallado en texto"""
        df = self.calculate_efficiency()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE BENCHMARK - MULTIPLICACIÓN DE MATRICES PARALELA\n")
            f.write("Big Data - Universidad de Las Palmas de Gran Canaria\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Archivo de datos: {self.csv_file}\n")
            f.write(f"Total de pruebas: {len(df)}\n")
            f.write(f"Tamaños probados: {sorted(df['size'].unique())}\n")
            f.write(f"Métodos probados: {len(df['method'].unique())}\n\n")
            
            # Resultados por tamaño
            f.write("RESULTADOS DETALLADOS POR TAMAÑO\n")
            f.write("-"*80 + "\n\n")
            
            for size in sorted(df['size'].unique()):
                f.write(f"\n{'='*80}\n")
                f.write(f"MATRICES {size}x{size}\n")
                f.write(f"{'='*80}\n\n")
                
                size_data = df[df['size'] == size].sort_values('execution_time')
                
                for _, row in size_data.iterrows():
                    f.write(f"Método: {row['method']}\n")
                    f.write(f"  Tiempo: {row['execution_time']:.4f}s\n")
                    f.write(f"  Memoria: {row['memory_used']:.1f}MB\n")
                    
                    if pd.notna(row.get('speedup')):
                        f.write(f"  Speedup: {row['speedup']:.2f}x\n")
                    
                    if pd.notna(row.get('efficiency')):
                        f.write(f"  Eficiencia: {row['efficiency']:.2%}\n")
                    
                    f.write("\n")
                
                # Mejor método para este tamaño
                best = size_data.iloc[0]
                f.write(f"🏆 Mejor: {best['method']} ({best['execution_time']:.4f}s)\n")
            
            # Resumen general
            f.write(f"\n{'='*80}\n")
            f.write("RESUMEN GENERAL\n")
            f.write(f"{'='*80}\n\n")
            
            avg_times = df.groupby('method')['execution_time'].mean().sort_values()
            f.write("Tiempo promedio por método:\n")
            for method, time in avg_times.items():
                f.write(f"  {method}: {time:.4f}s\n")
            
            f.write(f"\n🏆 Método más rápido (promedio): {avg_times.index[0]}\n")
            f.write(f"   Tiempo: {avg_times.values[0]:.4f}s\n")
            
            # Mejor speedup
            if 'speedup' in df.columns and df['speedup'].notna().any():
                max_speedup_idx = df['speedup'].idxmax()
                best_speedup = df.loc[max_speedup_idx]
                f.write(f"\nMejor speedup alcanzado:\n")
                f.write(f"  Método: {best_speedup['method']}\n")
                f.write(f"  Tamaño: {best_speedup['size']}x{best_speedup['size']}\n")
                f.write(f"  Speedup: {best_speedup['speedup']:.2f}x\n")
        
        print(f"\n✓ Reporte guardado en: {output_file}")


def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py <archivo_csv>")
        print("\nBuscando archivos CSV recientes...")
        
        csv_files = sorted(Path('.').glob('benchmark_results*.csv'), reverse=True)
        if csv_files:
            csv_file = str(csv_files[0])
            print(f"✓ Usando: {csv_file}")
        else:
            print("✗ No se encontraron archivos CSV")
            print("  Primero ejecuta: python benchmark_runner.py")
            return
    else:
        csv_file = sys.argv[1]
    
    print("\n" + "="*70)
    print("ANÁLISIS DE RESULTADOS - BENCHMARK MATRICES")
    print("="*70 + "\n")
    
    # Crear analizador
    analyzer = ResultsAnalyzer(csv_file)
    
    # Generar todos los análisis
    analyzer.generate_all_plots()
    analyzer.generate_report()
    
    print("\n✓ Análisis completado!")
    print("\nArchivos generados:")
    print("  - plots/01_execution_time.png")
    print("  - plots/02_speedup.png")
    print("  - plots/03_efficiency.png")
    print("  - plots/04_heatmap.png")
    print("  - plots/05_memory_usage.png")
    print("  - benchmark_report.txt")


if __name__ == "__main__":
    main()