from parser import build_dag
from dag import print_dag
from rules import apply_eval, apply_distrib_right, apply_distrib_left, available_actions

dag = build_dag("5+(2-3)*4")
print_dag(dag)
print(f"\nAvailable actions: {available_actions(dag)}")

plus_id  = dag.ops[0].node_id
times_id = dag.ops[1].node_id

print("\n--- DistribRight at + ---")
print_dag(apply_distrib_right(dag, plus_id))

print("\n--- DistribLeft at × ---")
print_dag(apply_distrib_left(dag, times_id))