"""io — HFSS 数据读写 + 优化结果存取。"""

from .hfss import read_hfss_csv, read_hfss_csv_db, read_hfss_dir, load_aep_patterns, compute_total_phases
from .result import ScenarioResult, load_elements, print_component_summary, save_result_figures
