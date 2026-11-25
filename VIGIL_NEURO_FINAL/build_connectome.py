"""Build two candidate induced graphs from public FAFB v783 release tables."""
import argparse
import base64
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from extract_graph import extract


TYPES = {"escape": {"LC4", "LPLC2", "DNp01"},
         "compass": {"EPG", "PEG", "PEN_a/PEN1", "PEN_b/PEN2", "Delta7"}}
PREFIX = "codex/data/fafb/783/"


def rows(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def build(data, out):
    catalog = {r["name"]: r for r in json.loads((data / "bucket.json").read_text())["items"]}
    products = {}
    names = ["consolidated_cell_types", "classification", "neurons", "connections_princeton"]
    for name in names:
        filename = name + ".csv.gz"
        path = data / filename
        entry = catalog[PREFIX + filename]
        with path.open("rb") as stream:
            check = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
        if check != entry["md5Hash"] or path.stat().st_size != int(entry["size"]):
            raise ValueError(f"Release integrity mismatch for {filename}")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        products[name] = dict(url="https://storage.googleapis.com/flywire-data/" + entry["name"],
                              sha256=sha, generation=entry["generation"], bytes=path.stat().st_size)
    selected = {r["root_id"]: r["primary_type"] for r in rows(data / (names[0]+".csv.gz"))
                if any(r["primary_type"] in types for types in TYPES.values())}
    side = {r["root_id"]: r["side"] for r in rows(data / "classification.csv.gz") if r["root_id"] in selected}
    nt = {r["root_id"]: (r["nt_type"], r["nt_type_score"]) for r in rows(data / "neurons.csv.gz")
          if r["root_id"] in selected}
    out.mkdir(parents=True, exist_ok=True)
    audit = {}
    for group, types in TYPES.items():
        folder = out / group
        folder.mkdir(exist_ok=True)
        ids = sorted(i for i,t in selected.items() if t in types)
        counts = Counter(selected[i] for i in ids)
        if set(counts) != types:
            raise ValueError(f"Missing expected annotation in {group}: {types-set(counts)}")
        id_file = folder / "ids.txt"
        id_file.write_text("\n".join(ids)+"\n", encoding="ascii")
        with (folder/"annotations.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["root_id","primary_type","side","predicted_nt","prediction_score"])
            for i in ids:
                w.writerow([i,selected[i],side.get(i,""),*nt.get(i,("",""))])
        manifest=extract(id_file,data/"connections_princeton.csv.gz",folder,"pre_root_id","post_root_id","syn_count",5,"FAFB-v783")
        audit[group]={"cell_types":dict(counts),**manifest}
    audit.update(products=products, attribution="FlyWire Consortium, FAFB v783; Princeton synapse predictions",
                 terms="https://join.flywire.ai/guidelines", upstream="Thresholded connection product, pooled threshold >=5. Weak-edge sensitivity requires the unfiltered table.",
                 scope="Exact primary-type selection. Predicted transmitters are metadata. Physiological signs, compass columns and circuit responses require validation.")
    (out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    build(args.data,args.out)
