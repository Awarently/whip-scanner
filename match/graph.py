"""Network graph of providers, owners/officials and addresses (networkx).

Bipartite graph: provider nodes ("P", row index) joined to the person/organisation nodes ("N", key) and
address-and-suite nodes ("A", key) they share with at least one other provider. Chain/corporate owners and
virtual addresses are excluded upstream (see match.network.build_edges). Connected components with two or
more providers are the clusters; degree and betweenness identify hub owners/addresses and bridge nodes.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd

BETWEENNESS_MAX_NODES = 400


def build_graph(edges: list[dict]) -> nx.Graph:
    g = nx.Graph()
    for e in edges:
        node = ("A" if e["type"] == "addr" else "N", e["key"])
        for p in (e["a"], e["b"]):
            g.add_node(("P", p), kind="provider")
            g.add_edge(("P", p), node)
        g.nodes[node]["kind"] = "address" if e["type"] == "addr" else "person"
    return g


def _label(node) -> str:
    kind, key = node
    if kind == "A":
        street, _, rest = key.partition("|")
        zip5, _, unit = rest.partition("|")
        return f"ADDRESS {street}{' STE ' + unit if unit else ''} {zip5}".strip()
    return ("ORG " if key.startswith("ORG:") else "PERSON ") + key.replace("ORG:", "")


def cluster_table(df: pd.DataFrame, edges: list[dict]):
    """Returns (df with cluster columns, clusters DataFrame). Clusters are ordered cross-sector first,
    then by size, then by Medicare payment."""
    g = build_graph(edges)
    df = df.copy()
    df["cluster_id"] = ""
    df["cluster_size"] = 1
    df["cluster_cross_sector"] = False
    df["cluster_hub"] = ""
    df["hub_degree"] = 0
    rows = []
    for comp in nx.connected_components(g):
        members = [n[1] for n in comp if n[0] == "P"]
        if len(members) < 2:
            continue
        sub = g.subgraph(comp)
        keys = [n for n in comp if n[0] != "P"]
        deg = {n: sub.degree(n) for n in keys}
        hub = sorted(keys, key=lambda n: (-deg[n], n[0] != "N", n[1]))[0]       # person before address on ties
        bridges = []
        if len(comp) <= BETWEENNESS_MAX_NODES:
            bt = nx.betweenness_centrality(sub)
            cut = set(nx.articulation_points(sub))
            bridges = [n for n in sorted(keys, key=lambda n: -bt[n]) if n in cut and n != hub][:3]
        sd = df.loc[members]
        sectors = sorted(sd["sector"].unique())
        rows.append({
            "members": members, "size": len(members), "sectors": "+".join(sectors),
            "cross_sector": len(sectors) > 1, "states": "+".join(sorted(sd["state"].unique())),
            "n_owner_nodes": sum(1 for n in keys if n[0] == "N"), "n_address_nodes": sum(1 for n in keys if n[0] == "A"),
            "hub_node": _label(hub), "hub_type": sub.nodes[hub]["kind"], "hub_degree": deg[hub],
            "bridge_nodes": "; ".join(f"{_label(n)} (betweenness {bt[n]:.2f})" for n in bridges),
            "link_types": "+".join(sorted({"addr" if n[0] == "A" else "person" for n in keys})),
            "link_keys": "; ".join(sorted(_label(n) for n in keys))[:500],
            "total_medicare_payment": float(sd["total_payment"].fillna(0).sum()),
            "provider_ids": "; ".join(sd["provider_id"]), "provider_names": "; ".join(sd["name"].astype(str))})
    rows.sort(key=lambda r: (not r["cross_sector"], -r["size"], -r["total_medicare_payment"]))
    out = []
    for k, r in enumerate(rows, 1):
        cid = f"C{k:04d}"
        m = r.pop("members")
        df.loc[m, "cluster_id"] = cid
        df.loc[m, "cluster_size"] = r["size"]
        df.loc[m, "cluster_cross_sector"] = r["cross_sector"]
        df.loc[m, "cluster_hub"] = r["hub_node"]
        df.loc[m, "hub_degree"] = r["hub_degree"]
        out.append({"cluster_id": cid, **r})
    cols = ["cluster_id", "size", "sectors", "cross_sector", "states", "hub_node", "hub_type", "hub_degree",
            "bridge_nodes", "n_owner_nodes", "n_address_nodes", "link_types", "total_medicare_payment",
            "provider_ids", "provider_names", "link_keys"]
    return df, pd.DataFrame(out, columns=cols)
