"""Makes fake network traffic records.

Real datasets like NSL-KDD or CIC-IDS are big, so to keep the repo light this
script creates something similar in spirit: normal traffic plus a small share
of attacks (port scans, floods and data exfiltration). If you have a real
dataset, save it as data/traffic.csv with numeric feature columns and a
`label` column (0 = normal, 1 = attack) and skip this step.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def normal_traffic(n):
    return pd.DataFrame({
        "duration": rng.exponential(2.0, n),
        "src_bytes": rng.lognormal(6.5, 1.0, n),
        "dst_bytes": rng.lognormal(7.0, 1.2, n),
        "packets": rng.poisson(20, n),
        "failed_logins": rng.binomial(1, 0.02, n),
        "conn_per_min": rng.poisson(5, n),
        "distinct_ports": rng.poisson(2, n) + 1,
        "label": 0,
    })


def attack_traffic(n):
    kinds = rng.choice(["scan", "flood", "exfil"], size=n)
    df = normal_traffic(n)
    for i, kind in enumerate(kinds):
        if kind == "scan":
            df.loc[i, "distinct_ports"] = rng.integers(15, 60)
            df.loc[i, "packets"] = rng.poisson(4)
            df.loc[i, "duration"] = rng.exponential(0.3)
        elif kind == "flood":
            df.loc[i, "conn_per_min"] = rng.poisson(80)
            df.loc[i, "packets"] = rng.poisson(300)
        else:
            df.loc[i, "src_bytes"] = rng.lognormal(11, 0.8)
            df.loc[i, "duration"] = rng.exponential(20.0)
    df["label"] = 1
    return df


def main(n_total=20000, attack_share=0.04):
    n_attack = int(n_total * attack_share)
    df = pd.concat([normal_traffic(n_total - n_attack), attack_traffic(n_attack)])
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # make it a bit messier so the attacks aren't trivially separable:
    # some normal records look odd (big backups, busy users)
    normal_rows = df[df["label"] == 0]
    odd = normal_rows.sample(frac=0.01, random_state=3).index
    df.loc[odd, "src_bytes"] *= 40

    df.to_csv("data/traffic.csv", index=False)
    print("saved", len(df), "rows,", df["label"].sum(), "attacks")


if __name__ == "__main__":
    main()
