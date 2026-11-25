"""Export anatomical adjacency and matched directed double-edge-swap controls."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scipy.sparse import csr_matrix, save_npz


def rewire(edges,seed):
    result=list(edges)
    occupied={(i,j) for i,j,w in result}
    if len(occupied)!=len(result):
        raise ValueError("Expected pooled, unique directed edges")
    movable=[k for k,(i,j,w) in enumerate(result) if i!=j]
    if len(movable)<2:
        raise ValueError("Insufficient swappable edges")
    rng=random.Random(seed)
    accepted=0
    for _ in range(200*len(movable)):
        a,b=rng.sample(movable,2)
        i,j,u=result[a]; k,l,v=result[b]
        if i==k or j==l or i==l or k==j or (i,l) in occupied or (k,j) in occupied:
            continue
        occupied.difference_update(((i,j),(k,l)))
        occupied.update(((i,l),(k,j)))
        result[a],result[b]=(i,l,u),(k,j,v)
        accepted+=1
        if accepted==10*len(movable):
            return result
    raise ValueError("Insufficient successful switches to build the requested control")


def export(graph,out,seeds):
    ids=(graph/"nodes.txt").read_text().splitlines()
    index={v:k for k,v in enumerate(ids)}
    with (graph/"edges.csv").open() as f:
        edges=[(index[r["pre_id"]],index[r["post_id"]],int(r["synapse_count"])) for r in csv.DictReader(f)]
    out.mkdir(parents=True,exist_ok=True)
    (out/"nodes.txt").write_text("\n".join(ids)+"\n")
    def matrix(values):
        pre,post,weights=zip(*values)
        return csr_matrix((weights,(post,pre)),shape=(len(ids),len(ids)))
    save_npz(out/"biological.npz",matrix(edges))
    natural={(i,j) for i,j,w in edges}
    controls={}
    for seed in seeds:
        values=rewire(edges,seed)
        assert Counter((i,w) for i,j,w in values)==Counter((i,w) for i,j,w in edges)
        assert Counter(j for i,j,w in values)==Counter(j for i,j,w in edges)
        assert len({(i,j) for i,j,w in values})==len(edges)
        save_npz(out/f"rewired-{seed}.npz",matrix(values))
        controls[seed]=len(natural.intersection((i,j) for i,j,w in values))/len(edges)
    report=dict(nodes=len(ids),edges=len(edges),orientation="row=postsynaptic, column=presynaptic",
                weights="Anatomical counts. Transmitter signs and physiological gains remain unspecified.",
                control="10 accepted switches per non-self edge; existing self edges fixed. Incoming and outgoing degree and each source's outgoing weight distribution preserved. Incoming strength and cell-type mixing may change.",
                upstream_edges_sha256=hashlib.sha256((graph/"edges.csv").read_bytes()).hexdigest(),
                fraction_original_edges_retained=controls)
    (out/"manifest.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[7,19,41])
    args=p.parse_args();print(export(args.graph,args.out,args.seeds))
