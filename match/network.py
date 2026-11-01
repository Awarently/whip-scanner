"""Link providers by shared address / owner / administrator / authorised official; clusters; churn."""
from __future__ import annotations
import itertools
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz
from clean.normalize import person_key, split_people, clean_str


def canonicalize_people(keys: set[str], thr: int) -> dict[str, str]:
    """Map near-identical person keys (same last name, same first initial) to one canonical key."""
    canon: dict[str, str] = {}
    blocks = defaultdict(list)
    for k in sorted(keys):
        if k.startswith("ORG:"):
            canon[k] = k
        else:
            blocks[k.split()[-1]].append(k)
    for last, ks in blocks.items():
        reps: list[str] = []
        for k in ks:
            for r in reps:
                if r[0] == k[0] and fuzz.ratio(r, k) >= thr:
                    canon[k] = r
                    break
            else:
                reps.append(k)
                canon[k] = k
    return canon


def add_people(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    roles = cfg["match"]["network_roles"]
    raw = []
    for _, r in df.iterrows():
        d = {}
        for role in roles:
            for nm in split_people(r.get(role, "") if pd.notna(r.get(role, "")) else ""):
                k = person_key(nm)
                if k:
                    d.setdefault(k, set()).add(role)
        raw.append(d)
    canon = canonicalize_people({k for d in raw for k in d}, cfg["match"]["person_threshold"])
    out = []
    for d in raw:
        m: dict[str, set] = {}
        for k, rs in d.items():
            m.setdefault(canon[k], set()).update(rs)
        out.append(m)
    df = df.copy()
    df["people"] = out
    return df


def build_edges(df: pd.DataFrame, cfg: dict, log=None):
    """Returns (edges, chain_edges). A person/organisation tied to more than `small_network_max` providers
    is a chain/corporate parent: its links go to chain_edges (informational, low weight) and are NOT used for
    shared-owner flags, clusters, churn or known-case linking. Address links are never treated as chain."""
    maxg = cfg["match"]["max_group_size"]
    small = cfg["match"]["small_network_max"]
    idx = list(df.index)
    groups: dict[tuple, list] = defaultdict(list)
    for i in idx:
        r = df.loc[i]
        if r["addr_key"] and not r["is_virtual"]:
            groups[("addr", r["addr_key"])].append(i)
        for k in r["people"]:
            groups[("person", k)].append(i)
    edges, chain_edges, skipped, chains = [], [], 0, 0
    for (typ, key), members in groups.items():
        if len(members) < 2:
            continue
        if len(members) > maxg:
            skipped += 1
            for m in members:                       # still count them as chain members
                chain_edges.append({"a": m, "b": m, "type": "chain_person", "key": key, "size": len(members)})
            continue
        if typ == "person" and len(members) > small:
            chains += 1
            for m in members:
                chain_edges.append({"a": m, "b": m, "type": "chain_person", "key": key, "size": len(members)})
            continue
        for a, b in itertools.combinations(members, 2):
            edges.append({"a": a, "b": b, "type": typ, "key": key})
    if log:
        log.info(f"network: {chains} chain/corporate owners (> {small} providers) and {skipped} very large "
                 f"keys (> {maxg}) treated as chain ownership, excluded from shared-owner flags and clusters")
    return edges, chain_edges


def annotate(df: pd.DataFrame, cfg: dict, log=None):
    """Adds link-count columns, churn flags and cluster ids. Returns (df, edges, clusters_df)."""
    df = add_people(df, cfg)
    edges, chain_edges = build_edges(df, cfg, log)
    n = len(df)
    pos = {i: p for p, i in enumerate(df.index)}
    sector = df["sector"].to_dict()
    cols = {c: [0] * n for c in ("n_same_addr_same_sector", "n_same_addr_all",
                                 "n_shared_people_same_sector", "n_shared_people_all")}
    partners = defaultdict(lambda: defaultdict(set))
    for e in edges:
        for x, y in ((e["a"], e["b"]), (e["b"], e["a"])):
            partners[x][e["type"]].add(y)
    for i in df.index:
        a = partners[i]["addr"]
        p = partners[i]["person"]
        cols["n_same_addr_all"][pos[i]] = len(a)
        cols["n_same_addr_same_sector"][pos[i]] = sum(1 for j in a if sector[j] == sector[i])
        cols["n_shared_people_all"][pos[i]] = len(p)
        cols["n_shared_people_same_sector"][pos[i]] = sum(1 for j in p if sector[j] == sector[i])
    for c, v in cols.items():
        df[c] = v
    ids = df["provider_id"].to_dict()
    df["link_detail"] = [
        "; ".join([f"addr-shared with {', '.join(sorted(ids[j] for j in partners[i]['addr']))}" if partners[i]["addr"] else "",
                   f"person-shared with {', '.join(sorted(ids[j] for j in partners[i]['person']))}" if partners[i]["person"] else ""]).strip("; ")
        for i in df.index]

    chain = defaultdict(dict)
    for e in chain_edges:
        chain[e["a"]][e["key"]] = e["size"]
    df["chain_owner_count"] = [len(chain[i]) for i in df.index]
    df["chain_owner_detail"] = ["; ".join(f"{k.replace('ORG:', '')} ({n} providers)"
                                          for k, n in sorted(chain[i].items(), key=lambda kv: -kv[1])[:5])
                                for i in df.index]

    # churn: closed provider, successor licensed within N months, same sector, shared owner or address
    months = cfg["match"]["churn_months"]
    df["license_churn"] = False
    df["churn_detail"] = ""
    for e in edges:
        a, b = e["a"], e["b"]
        if sector[a] != sector[b]:
            continue
        for old, new in ((a, b), (b, a)):
            cd, ld = df.at[old, "close_date"], df.at[new, "license_date"]
            if pd.notna(cd) and pd.notna(ld) and cd <= ld <= cd + pd.DateOffset(months=months):
                msg = (f"{ids[old]} closed {cd:%Y-%m-%d}; {ids[new]} licensed {ld:%Y-%m-%d} "
                       f"(shared {e['type']}: {e['key']})")
                for x in (old, new):
                    df.at[x, "license_churn"] = True
                    df.at[x, "churn_detail"] = (df.at[x, "churn_detail"] + " | " + msg).strip(" |")

    # clusters (connected components)
    parent = {i: i for i in df.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        parent[find(e["a"])] = find(e["b"])
    comp = defaultdict(list)
    for i in df.index:
        comp[find(i)].append(i)
    df["cluster_id"] = ""
    df["cluster_size"] = 1
    df["cluster_cross_sector"] = False
    rows, k = [], 0
    for _, members in sorted(comp.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        k += 1
        cid = f"C{k:04d}"
        sub = df.loc[members]
        sectors = sorted(sub["sector"].unique())
        links = [e for e in edges if e["a"] in members]
        df.loc[members, "cluster_id"] = cid
        df.loc[members, "cluster_size"] = len(members)
        df.loc[members, "cluster_cross_sector"] = len(sectors) > 1
        rows.append({"cluster_id": cid, "size": len(members), "sectors": "+".join(sectors),
                     "cross_sector": len(sectors) > 1, "states": "+".join(sorted(sub["state"].unique())),
                     "provider_ids": "; ".join(sub["provider_id"]),
                     "provider_names": "; ".join(sub["name"].astype(str)),
                     "link_types": "+".join(sorted({e["type"] for e in links})),
                     "link_keys": "; ".join(sorted({f"{e['type']}:{e['key']}" for e in links}))[:500],
                     "total_medicare_payment": float(sub["total_payment"].fillna(0).sum())})
    clusters = pd.DataFrame(rows)
    return df, edges, clusters
