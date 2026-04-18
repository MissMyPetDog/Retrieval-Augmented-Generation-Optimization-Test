"""
Three-way RAG pipeline comparison:
  1. Pure BruteForce baseline  (run_bruteforce.py)
  2. IVF default / no deep optim (run_intermediate.py)
  3. Fully optimized (run_optimized.py)

Each module exposes a .run() function returning the same result schema.
See compare_pipelines.ipynb for the side-by-side comparison.
"""