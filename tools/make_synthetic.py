"""Synthetic FL data (fictional names) with injected red-flag patterns, for tests and --demo."""
from __future__ import annotations
import random
from pathlib import Path
import pandas as pd


def generate(root: Path, seed: int = 7) -> dict:
    rnd = random.Random(seed)
    m = root / "manual"
    (m / "FL" / "home_health").mkdir(parents=True, exist_ok=True)
    (m / "FL" / "hospice").mkdir(parents=True, exist_ok=True)
    (m / "shared").mkdir(parents=True, exist_ok=True)
    ids = {}

    firsts = ["Alan", "Beth", "Carl", "Dina", "Evan", "Fran", "Gus", "Hana", "Ivan", "Jude", "Kara", "Liam"]
    lasts = ["Arbor", "Bell", "Crane", "Dunn", "Ellis", "Frost", "Grove", "Hale", "Irwin", "Joyce", "Knoll", "Lowe",
             "Marsh", "Nash", "Orr", "Pike", "Quade", "Rowe", "Sloan", "Thorne", "Usher", "Vale", "Wynn", "Yates"]
    pool = [f"{f} {l}" for f in firsts for l in lasts]
    rnd.shuffle(pool)

    def person():
        return pool.pop()     # unique, so no accidental shared owners

    # ---------------- home health
    lic, prov, util = [], [], []

    def add_hh(key, name, street, zip_, county, lic_date, owner, ccn, ppb, bene, status="Active", close="",
               sanction="", insp=0, npi="", admin="", eps=1.8):
        lic.append({"Provider Name": name, "License Number": f"HHA{ccn[-4:]}", "Address Line 1": street, "City": "Testville",
                    "Zip": zip_, "County": county, "License Date": lic_date, "Status": status, "Close Date": close,
                    "Owner": owner, "Administrator": admin, "Sanction": sanction, "Failed Inspections": insp})
        prov.append({"CMS Certification Number (CCN)": ccn, "Provider Name": name.upper().replace("LLC", ", L.L.C."),
                     "Address Line 1": street.upper(), "ZIP Code": zip_, "NPI": npi, "Certification Date": lic_date})
        if bene:
            util.append({"CCN": ccn, "Total Beneficiaries": bene, "Total Episodes": round(bene * eps),
                         "Total Medicare Payment": round(ppb * bene), "Year": 2024})
        ids[key] = f"FL-HH-{ccn}"

    n = 0
    for i in range(60):
        n += 1
        add_hh(f"hh{i}", f"Sample Home Health {i:02d} LLC", f"{100 + i * 7} Oak Avenue", "33101" if i % 2 else "33102",
               "ALPHA" if i % 2 else "BETA", f"20{rnd.randint(10, 21)}-0{rnd.randint(1, 9)}-15",
               "Chain Parent Holdings LLC" if i < 8 else person(), f"10{n:04d}", rnd.gauss(3000, 300), rnd.randint(80, 400), eps=rnd.gauss(1.8, 0.1))
    base = 70
    add_hh("addrA", "Suite Twin One LLC", "100 Main Street Suite 200", "33101", "ALPHA", "2015-03-01", person(), "200001", 3000, 150)
    add_hh("addrB", "Suite Twin Two LLC", "100 Main St., Ste 200", "33101", "ALPHA", "2016-03-01", person(), "200002", 3000, 150)
    add_hh("ownC", "Owner Share Alpha LLC", "11 Elm Road", "33102", "BETA", "2014-05-01", "Zed Quill", "200003", 3000, 150)
    add_hh("ownD", "Owner Share Beta LLC", "900 Pine Lane", "33101", "ALPHA", "2012-05-01", "Quill, Zed", "200004", 3000, 150)
    add_hh("newE", "Brand New Fast Billing LLC", "5 Birch Court", "33101", "ALPHA", "2026-02-01", person(), "200005", 6500, 1200, eps=2.0)
    add_hh("topF", "Extreme Payment Outlier LLC", "77 Cedar Drive", "33102", "BETA", "2010-02-01", person(), "200006", 19000, 300, eps=5.0)
    add_hh("closedG", "Old Agency Closed LLC", "40 Fir Street", "33101", "ALPHA", "2012-01-01", "Reopen Owner", "200007", 3000, 150,
           status="Closed", close="2026-01-15")
    add_hh("churnH", "New Agency Reopened LLC", "41 Fir Street", "33101", "ALPHA", "2026-06-01", "Reopen Owner", "200008", 3000, 150)
    add_hh("virtI", "Mailbox Agency LLC", "PO Box 4412", "33102", "BETA", "2013-01-01", person(), "200009", 3000, 150)
    add_hh("exclJ", "Owner Excluded Match LLC", "8 Spruce Way", "33102", "BETA", "2011-01-01", "Persn Excluded", "200010", 3000, 150)
    add_hh("knownK", "Already Convicted Agency LLC", "300 Maple Avenue Suite 5", "33101", "ALPHA", "2012-01-01", person(), "200011", 9000, 300,
           npi="1999999991")
    add_hh("linkL", "Neighbor Of Known LLC", "300 Maple Ave #5", "33101", "ALPHA", "2018-01-01", person(), "200012", 3000, 150)
    add_hh("pressN", "Press Linked Owner LLC", "22 Ash Street", "33102", "BETA", "2012-01-01", "Pressnamed Defendant", "200013", 3000, 150)
    add_hh("crossM", "Cross Sector Home Health LLC", "60 Walnut Road", "33101", "ALPHA", "2012-01-01", "Cross Sector Owner", "200014", 3000, 150)
    add_hh("sanctO", "Sanctioned Agency LLC", "61 Walnut Road", "33101", "ALPHA", "2012-01-01", person(), "200015", 3000, 150, sanction="Y", insp=3)
    pd.DataFrame(lic).to_csv(m / "FL/home_health/license.csv", index=False)
    pd.DataFrame(prov).to_csv(m / "FL/home_health/cms_provider.csv", index=False)
    pd.DataFrame(util).to_csv(m / "FL/home_health/cms_util.csv", index=False)

    # ---------------- hospice
    hl, hp, hu, hq = [], [], [], []

    def add_ho(key, name, street, zip_, county, lic_date, owner, ccn, bene, pay_pb, los, live, nc, cap, fin, cap_years=0):
        hl.append({"Facility Name": name, "License Number": f"HOS{ccn[-4:]}", "Address Line 1": street, "City": "Testville",
                   "Zip": zip_, "County": county, "License Date": lic_date, "Status": "Active", "Owner": owner})
        hp.append({"CMS Certification Number (CCN)": ccn, "Facility Name": name.upper(), "Address Line 1": street.upper(),
                   "ZIP Code": zip_, "Certification Date": lic_date})
        hu.append({"CCN": ccn, "Total Beneficiaries": bene, "Total Medicare Payment": round(pay_pb * bene),
                   "Average Length of Stay": los, "Live Discharge Rate": live, "Non Cancer Share": nc,
                   "Cap Utilization": cap, "Years Over Cap": cap_years, "Year": 2024})
        hq.append({"CCN": ccn, "Visits In Last Days Of Life": fin})
        ids[key] = f"FL-HO-{ccn}"

    for i in range(40):
        add_ho(f"ho{i}", f"Sample Hospice {i:02d} LLC", f"{500 + i * 11} Lake Boulevard", "33103" if i % 2 else "33104",
               "GAMMA" if i % 2 else "DELTA", f"20{rnd.randint(8, 20)}-04-10", person(), f"50{i:04d}",
               rnd.randint(100, 500), rnd.gauss(12000, 1000), rnd.gauss(85, 8), rnd.gauss(12, 2), rnd.gauss(55, 5),
               rnd.gauss(60, 8), rnd.gauss(70, 5))
    add_ho("losP", "Long Stay Hospice LLC", "1 River Road", "33103", "GAMMA", "2012-01-01", person(), "600001", 300, 30000, 400, 30, 90, 99, 40, 3)
    add_ho("newQ", "New Hospice Big Billing LLC", "2 River Road", "33104", "DELTA", "2026-03-01", person(), "600002", 900, 22000, 150, 20, 80, 80, 60)
    add_ho("crossM", "Cross Sector Hospice LLC", "3 River Road", "33104", "DELTA", "2012-01-01", "Cross Sector Owner", "600003", 200, 12000, 80, 12, 55, 60, 70)
    pd.DataFrame(hl).to_csv(m / "FL/hospice/license.csv", index=False)
    pd.DataFrame(hp).to_csv(m / "FL/hospice/cms_provider.csv", index=False)
    pd.DataFrame(hu).to_csv(m / "FL/hospice/cms_util.csv", index=False)
    pd.DataFrame(hq).to_csv(m / "FL/hospice/cms_quality.csv", index=False)

    # ---------------- shared
    pd.DataFrame([{"LASTNAME": "EXCLUDED", "FIRSTNAME": "PERSON", "MIDNAME": "", "BUSNAME": "", "NPI": "0000000000", "STATE": "FL", "EXCLTYPE": "1128a1"},
                  {"LASTNAME": "", "FIRSTNAME": "", "MIDNAME": "", "BUSNAME": "UNRELATED EXCLUDED CO", "NPI": "1999999991", "STATE": "FL", "EXCLTYPE": "1128b7"},
                  ]).to_csv(m / "shared/leie.csv", index=False)
    pd.DataFrame([{"date": "2026-05-01", "title": "Fictional press release", "url": "https://example.invalid/pr1",
                   "names": "Pressnamed Defendant; Some Other Defendant", "state": "FL"}]).to_csv(m / "shared/press_releases.csv", index=False)
    pd.DataFrame([{"entity_name": "Owner Share Alpha LLC", "formation_date": "2014-04-20", "status": "ACTIVE",
                   "officer_name": "Zed Quill", "officer_title": "MGR", "registered_agent": "Generic Agent Services"},
                  {"entity_name": "Owner Share Alpha LLC", "formation_date": "2014-04-20", "status": "ACTIVE",
                   "officer_name": "Generic Agent Services", "officer_title": "REGISTERED AGENT", "registered_agent": "Generic Agent Services"}
                  ]).to_csv(m / "FL/corp.csv", index=False)
    _ga(m, ids, pool, rnd)
    return ids


def _ga(m: Path, ids: dict, pool: list, rnd: random.Random) -> None:
    """Georgia: HHA export + Private Home Care Provider export (two files), plus a hospice license list."""
    gh = m / "GA" / "home_health"
    gh.mkdir(parents=True, exist_ok=True)
    (m / "GA" / "hospice").mkdir(parents=True, exist_ok=True)
    cols = ["Facility Name", "License Number", "Address", "City", "Zip", "County", "Original License Date", "Status", "Owner", "Administrator"]

    def row(name, lic, street, zip_, county, date, owner, status="Active", admin=""):
        return [name, lic, street, "Gatown", zip_, county, date, status, owner, admin]
    hha = [row(f"Ga Sample HHA {i:02d} Inc", f"GAH{i:04d}", f"{10 + i * 3} Peach Street", "30301", "FULTON", "2015-06-01", pool.pop(), admin=pool.pop())
           for i in range(15)]
    phcp = [row(f"Ga Sample Home Care {i:02d} LLC", f"GAP{i:04d}", f"{400 + i * 5} Pine Road", "30302", "FULTON", "2016-01-01", pool.pop())
            for i in range(15)]
    # shared suite across the two license types, shared owner across the two types, virtual address
    hha.append(row("Ga Suite Share HHA Inc", "GAH9001", "500 Peachtree Street, Suite 12", "30303", "FULTON", "2014-01-01", pool.pop()))
    phcp.append(row("Ga Suite Share Home Care LLC", "GAP9001", "500 Peachtree St Ste 12", "30303", "FULTON", "2019-01-01", pool.pop()))
    hha.append(row("Ga Owner Share HHA Inc", "GAH9002", "1 Magnolia Way", "30304", "DEKALB", "2013-01-01", "Gail Ravenel"))
    phcp.append(row("Ga Owner Share Home Care LLC", "GAP9002", "2 Magnolia Way", "30304", "DEKALB", "2020-01-01", "Ravenel, Gail"))
    phcp.append(row("Ga Mailbox Home Care LLC", "GAP9003", "PMB 220, 77 Elm Street", "30305", "FULTON", "2021-01-01", pool.pop()))
    pd.DataFrame(hha, columns=cols).to_csv(gh / "license_hha.csv", index=False)
    pd.DataFrame(phcp, columns=cols).to_csv(gh / "license_phcp.csv", index=False)
    ho = [row(f"Ga Sample Hospice {i:02d} Inc", f"GAO{i:04d}", f"{900 + i * 4} Cherry Lane", "30306", "FULTON", "2012-01-01", pool.pop()) for i in range(12)]
    ho.append(row("Ga Hospice Owner Share Inc", "GAO9001", "9 Dogwood Drive", "30307", "FULTON", "2018-01-01", "Gail Ravenel"))
    pd.DataFrame(ho, columns=cols).to_csv(m / "GA" / "hospice" / "license.csv", index=False)
