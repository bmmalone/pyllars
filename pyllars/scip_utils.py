"""
Utilities for working with SCIP: http://scip.zib.de/
"""
import numpy as np

import pyllars.utils as utils
import toolz.dicttoolz

###
# Parsing output logs
###

def _parse_status_line(l):
    # SCIP Status        : problem is solved [optimal solution found]
    
    was_solved = l.find("problem is solved") > -1
    time_out = l.find("time limit reached") > -1
    mem_out = l.find("memory limit reached") > -1
    
    return {
        "was_solved": was_solved,
        "time_out": time_out,
        "mem_out": mem_out,
        "has_error": False
    }

def _parse_solving_time_line(l):
    # Solving Time (sec) : 0.15
    
    s = l.split(":")
    solving_time = float(s[1].strip())
    return {"solving_time": solving_time}

def _parse_solving_nodes_line(l):
    # Solving Nodes      : 1
    
    s = l.split(":")
    s = s[1].split()
    nodes = int(s[0])
    return {"nodes": nodes}



def _parse_primal_bound_line(l):
    # Primal Bound       : +6.20270000000000e+04 (13 solutions)
    # Primal Bound       : -1.00000000000000e+20 (0 solutions)
    
    s = l.split(":")
    s = s[1].split()
    primal_bound = float(s[0])
    
    # number of solutions is the second token after the colon,
    # but it starts with a "(", so remove it
    num_solutions = int(s[1][1:])
    
    return {"primal_bound": primal_bound, "num_solutions": num_solutions}

def _parse_dual_bound_line(l):
    # Dual Bound         : +6.20270000000000e+04
    # Dual Bound         : +0.00000000000000e+00
    
    s = l.split(":")
    dual_bound = float(s[1].strip())
    return {"dual_bound": dual_bound}

def _parse_gap_line(l):
    # Gap                : 0.00 %
    # Gap                : infinite
    
    s = l.split(":")
    s = s[1].split()
    gap = utils.try_parse_float(s[0])
    
    if gap is None:
        gap = np.inf
    
    return {"gap": gap}
    

STARTS_WITH_MAP = {
    "SCIP Status": _parse_status_line,
    "Solving Time (sec)": _parse_solving_time_line,
    "Solving Nodes": _parse_solving_nodes_line,
    "Primal Bound": _parse_primal_bound_line,
    "Dual Bound": _parse_dual_bound_line,
    "Gap": _parse_gap_line
}


def get_solving_status(log_file):
    # initially, assume some error happened
    ret = {
        'was_solved': False,
        'time_out': False,
        'mem_out': False,
        'has_error': True,
        'primal_bound': np.inf,
        'dual_bound': -np.inf,
        'gap': np.inf,
        'num_solutions': 0
    }
    ret = [ret]

    with open(log_file) as lf:
        for line in lf:
            for sw, f in STARTS_WITH_MAP.items():
                if line.startswith(sw):
                    ret.append(f(line))

    # if we found a valid status line, then that will overwrite the initial
    # error entries since "[l]ater dictionaries have precedence"
    #
    # http://toolz.readthedocs.io/en/latest/api.html#toolz.dicttoolz.merge
    ret = toolz.dicttoolz.merge(*ret)
    ret['log_file'] = log_file
    ret['problem'] = utils.get_basename(log_file)
    return ret

