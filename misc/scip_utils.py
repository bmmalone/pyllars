"""
Utilities for working with SCIP: http://scip.zib.de/
"""

import misc.utils as utils
import toolz.dicttoolz

###
# Parsing output logs
###

def _parse_status_line(l):
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
    s = l.split(":")
    solving_time = float(s[1].strip())
    return {"solving_time": solving_time}

def _parse_solving_nodes_line(l):
    s = l.split(":")
    s = s[1].split()
    nodes = int(s[0])
    return {"nodes": nodes}

STARTS_WITH_MAP = {
    "SCIP Status": _parse_status_line,
    "Solving Time (sec)": _parse_solving_time_line,
    "Solving Nodes": _parse_solving_nodes_line
}

def get_solving_status(log_file):
    # initially, assume some error happened
    ret = {
        'was_solved': False,
        'time_out': False,
        'mem_out': False,
        'has_error': True,
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

