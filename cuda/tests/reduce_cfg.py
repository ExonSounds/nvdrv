#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# usage: reduce_cfg.py dump.cfg build/cfgtest > minimal.cfg   (keeps shapes that hit route_to)
"""Shrink a CFG (cfgtest format) while cfgtest still fails (exit != 0 and != 3 means assertion; 3 = unroutable).
A node is removed by redirecting its predecessors to one of its successors (the first), unless it is 0."""
import re, subprocess, sys, random
src, tool = sys.argv[1], sys.argv[2]
succ = {}; end = None
for line in open(src):
    m = re.match(r'cfg (\d+): goto_if (\d+) (\d+)', line)
    if m: succ[int(m[1])] = [int(m[2]), int(m[3])]; continue
    m = re.match(r'cfg (\d+): goto (\d+)', line)
    if m: succ[int(m[1])] = [int(m[2])]; continue
    m = re.match(r'cfg end (\d+)', line)
    if m: end = int(m[1])

def write(g, path):
    with open(path, 'w') as f:
        for b in sorted(g):
            s = g[b]
            f.write(f"cfg {b}: goto_if {s[0]} {s[1]}\n" if len(s) == 2 else f"cfg {b}: goto {s[0]}\n")
        f.write(f"cfg end {end}\n")

def renumber(g):
    # keep node 0 and end; others get compact ids
    ids = {0: 0}; nxt = 1
    for b in sorted(g):
        if b not in ids and b != end: ids[b] = nxt; nxt += 1
    ids[end] = nxt
    ng = {ids[b]: [ids[t] for t in s] for b, s in g.items()}
    return ng, nxt

def fails(g):
    ng, e = renumber(g)
    global end_cur
    path = src + '.try'
    with open(path, 'w') as f:
        for b in sorted(ng):
            s = ng[b]
            f.write(f"cfg {b}: goto_if {s[0]} {s[1]}\n" if len(s) == 2 else f"cfg {b}: goto {s[0]}\n")
        f.write(f"cfg end {e}\n")
    r = subprocess.run([tool, path], capture_output=True, timeout=600)
    return b"route_to: unrouted" in r.stderr

def remove(g, b, which):
    if b == 0 or b == end or b not in g: return None
    t = g[b][which] if which < len(g[b]) else None
    if t is None or t == b: return None
    ng = {}
    for x, s in g.items():
        if x == b: continue
        ns = [t if y == b else y for y in s]
        if len(ns) == 2 and ns[0] == ns[1]: ns = [ns[0]]
        if len(ns) == 2 and end in ns: return None  # a goto_if to end_block is not a shape the front end makes
        ng[x] = ns
    return ng

assert fails(succ), "the input does not fail"
g = succ
changed = True
rounds = 0
while changed:
    changed = False
    rounds += 1
    for b in sorted(g, reverse=True):
        for which in (0, 1):
            ng = remove(g, b, which)
            if ng is None: continue
            if fails(ng):
                g = ng; changed = True
                print(f"removed {b} (via succ {which}) -> {len(g)} blocks", file=sys.stderr)
                break
    # also try dropping conditional edges (goto_if -> goto)
    for b in sorted(g, reverse=True):
        if len(g[b]) == 2:
            for keep in (0, 1):
                ng = dict(g); ng[b] = [g[b][keep]]
                if fails(ng):
                    g = ng; changed = True; break
ng, e = renumber(g)
for b in sorted(ng):
    s = ng[b]
    print(f"cfg {b}: goto_if {s[0]} {s[1]}" if len(s) == 2 else f"cfg {b}: goto {s[0]}")
print(f"cfg end {e}")
