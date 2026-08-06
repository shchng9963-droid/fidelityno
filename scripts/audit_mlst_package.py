from pathlib import Path
import re

root = Path("manuscript/mlst")
main = (root / "main.tex").read_text()
supp = (root / "supplementary.tex").read_text()
bib = (root / "bibliography.bib").read_text()
errors = []
label_pattern = re.compile(r"\\label\{([^}]+)\}")

for name, source in [("main", main), ("supplementary", supp)]:
    labels = set(label_pattern.findall(source))
    refs = set()
    for match in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", source):
        refs.update(part.strip() for part in match.split(",") if not part.strip().startswith("#"))
    missing_refs = sorted(refs - labels)
    if missing_refs:
        errors.append(f"{name}: missing labels {missing_refs}")

all_text = main + "\n" + supp
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
cite_keys = set()
for match in re.findall(r"\\cite\{([^}]+)\}", all_text):
    cite_keys.update(part.strip() for part in match.split(",") if not part.strip().startswith("#"))
missing_cites = sorted(cite_keys - bib_keys)
if missing_cites:
    errors.append(f"missing bibliography keys {missing_cites}")

figures = []
for name, source in [("main", main), ("supplementary", supp)]:
    for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        candidate = Path(fig)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".pdf")
        path = root / candidate
        if not path.is_file():
            path = root / "figures" / candidate
        figures.append((name, path))
        if not path.is_file():
            errors.append(f"{name}: missing figure {candidate}")

for pattern in [
    r"6000", r"6,000", r"76 unit", r"five collision", r"five families",
    r"exact Choi composition\. In an experimental", r"rather than the \$0\.096\$ reported there",
]:
    if re.search(pattern, all_text, re.I):
        errors.append(f"stale claim matched: {pattern}")

abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
abstract_words = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", abstract_match.group(1))) if abstract_match else 0
print(f"main labels: {len(set(label_pattern.findall(main)))}")
print(f"supplement labels: {len(set(label_pattern.findall(supp)))}")
print(f"citations used / bib entries: {len(cite_keys)} / {len(bib_keys)}")
print(f"abstract words (approx.): {abstract_words}")
print("figures:")
for name, path in figures:
    size = path.stat().st_size if path.is_file() else 0
    print(f"  {name}: {path.relative_to(root)} ({size} bytes)")
if errors:
    print("AUDIT FAILED")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)
print("AUDIT PASSED")
