#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Audit all dequant call sites for correct CQM (scaling list) indices.

Greps mb.rs for dequant_4x4/dequant_8x8 calls and extracts the CQM index
or table reference used. Reports each call site for manual review.

Expected CQM indices (H.264 4:2:0):
  0 = Intra Y 4x4        3 = Inter Y 4x4
  1 = Intra Cb 4x4       4 = Inter Cb 4x4
  2 = Intra Cr 4x4       5 = Inter Cr 4x4
  0 = Intra Y 8x8        3 = Inter Y 8x8

Usage:
    python3 scripts/audit_dequant_cqm.py
"""
import re
import sys

MB_RS = "codecs/wedeo-codec-h264/src/mb.rs"

# Patterns to find dequant calls and their table arguments
PATTERNS = [
    (r"dequant::dequant_4x4\(", "4x4"),
    (r"dequant::dequant_8x8\(", "8x8"),
    (r"dc_dequant_scale\(", "DC"),
]


def main():
    try:
        with open(MB_RS) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: {MB_RS} not found. Run from project root.")
        sys.exit(1)

    print(f"Auditing dequant call sites in {MB_RS}")
    print(f"{'Line':>5} {'Type':>4}  {'CQM Context':<50}  {'Surrounding Code'}")
    print("-" * 110)

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        for pattern, dq_type in PATTERNS:
            if re.search(pattern, stripped):
                # Extract CQM context from the table reference
                cqm_info = "?"

                # Look for coeffs[N] pattern in this line or next few lines
                context_lines = "".join(lines[i - 1 : i + 4])
                m = re.search(r"coeffs\[(\d+)\]", context_lines)
                if m:
                    idx = int(m.group(1))
                    names = {
                        0: "Intra Y",
                        1: "Intra Cb",
                        2: "Intra Cr",
                        3: "Inter Y",
                        4: "Inter Cb",
                        5: "Inter Cr",
                    }
                    cqm_info = f"CQM={idx} ({names.get(idx, '???')})"
                elif re.search(r"coeffs\[cqm\]", context_lines):
                    # Variable CQM, find its definition
                    for j in range(max(0, i - 10), i):
                        cm = re.search(r"let cqm\s*=\s*(.+?);", lines[j])
                        if cm:
                            cqm_info = f"CQM=var ({cm.group(1).strip()})"
                            break
                elif re.search(r"coeffs\[ac_cqm\]", context_lines):
                    for j in range(max(0, i - 10), i):
                        cm = re.search(r"let ac_cqm\s*=\s*(.+?);", lines[j])
                        if cm:
                            cqm_info = f"CQM=var ({cm.group(1).strip()})"
                            break
                elif re.search(r"coeffs\[dc_cqm\]", context_lines):
                    for j in range(max(0, i - 10), i):
                        cm = re.search(r"let dc_cqm\s*=\s*(.+?);", lines[j])
                        if cm:
                            cqm_info = f"CQM=var ({cm.group(1).strip()})"
                            break

                # Also look for dc_dequant_scale's list argument
                m2 = re.search(r"dc_dequant_scale\(&ctx\.dequant4,\s*(.+?),", stripped)
                if m2:
                    cqm_info = f"list={m2.group(1).strip()}"

                # Find enclosing function for context
                func = "?"
                for j in range(i - 1, max(0, i - 80), -1):
                    fm = re.search(r"fn\s+(\w+)", lines[j])
                    if fm:
                        func = fm.group(1)
                        break

                short_line = stripped[:60] + ("..." if len(stripped) > 60 else "")
                print(f"{i:>5} {dq_type:>4}  {cqm_info:<50}  {func}(): {short_line}")
                break

    print()
    print("Expected: Intra Y=0, Inter Y=3, Intra Cb=1, Intra Cr=2, Inter Cb=4, Inter Cr=5")


if __name__ == "__main__":
    main()
