from parser import build_dag
from dag import print_dag

root = build_dag("5+(2-3)*4")
print_dag(root)