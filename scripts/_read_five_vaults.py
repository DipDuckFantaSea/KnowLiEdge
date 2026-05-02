from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT / "sandbox" / "data" / "02_markdown_vault"
ITEMS = [
    ("S1", "1cda3b9e0bdd892c42eb6fd0cb781bad4bbf8237"),
    ("S4", "dfd76a887803eec9beda3cb7067200647c0be33e"),
    ("S8", "3888a55ea0fdc0d11dbc5d69e47368f0033f320b"),
    ("S16", "3b37a98373e52a1d87a866ca1b43f56dfa16610b"),
    ("S18", "2097adbf103a6b8d86c371c26f875b2f6a7157a1"),
]

def main() -> None:
    out = []
    for tag, did in ITEMS:
        p = VAULT / f"{did}.md"
        t = p.read_text(encoding="utf-8", errors="replace")
        out.append("=" * 72)
        out.append(f"{tag} {did}")
        out.append(t[:8000])
    Path(__file__).resolve().parent.parent.joinpath(".knotliedge", "five_vaults_v2.txt").write_text(
        "\n".join(out), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
