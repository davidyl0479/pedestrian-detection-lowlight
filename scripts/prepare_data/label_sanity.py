from pathlib import Path
import math
import sys

# --- Config ---
ALLOWED = {0, 1, 2}  # pedestrian, bicycledriver, motorbikedriver
ROOT = Path("data/raw/labels")  # your mirrored labels
SPLITS = ["train", "val"]  # check both
EPS = 1e-6  # tolerance around [0,1]; aligns with 6-decimal labels

errors = 0
warnings = 0


def err(msg):
    global errors
    print(f"ERROR: {msg}")
    errors += 1


def warn(msg):
    global warnings
    print(f"WARN:  {msg}")
    warnings += 1


def is_finite(*vals):
    return all(math.isfinite(v) for v in vals)


def check_range(name, v, lo=0.0, hi=1.0, path="", ln=0):
    """Range check with epsilon: hard error if < lo-EPS or > hi+EPS; warning if just outside."""
    if v < lo - EPS or v > hi + EPS:
        err(f"[RANGE] {path}:{ln} {name}={v:.8f} not in [{lo},{hi}] (|>EPS|)")
        return False
    if v < lo or v > hi:
        warn(f"[RANGE] {path}:{ln} {name}={v:.8f} slightly outside [{lo},{hi}] (<=EPS)")
    return True


def check_pos(name, v, lo=0.0, hi=1.0, path="", ln=0):
    """Positive (strict) with epsilon and upper bound [0,1]."""
    # lower bound
    if v <= lo - EPS:
        err(f"[SIZE]  {path}:{ln} {name}={v:.8f} must be > {lo}")
        return False
    if v <= lo:
        warn(f"[SIZE]  {path}:{ln} {name}={v:.8f} is ~0 (<=EPS)")
    # upper bound
    return check_range(name, v, lo, hi, path, ln)


def parse_class(raw, path, ln):
    """Accept '0' or '0.0' etc. Must be within EPS of an integer, then cast."""
    try:
        cf = float(raw)
    except Exception as e:
        err(f"[PARSE] {path}:{ln} class parse failed: {raw!r} ({e})")
        return None
    if not math.isfinite(cf):
        err(f"[PARSE] {path}:{ln} class is not finite: {raw!r}")
        return None
    rounded = round(cf)
    if abs(cf - rounded) > EPS:
        err(f"[CLASS] {path}:{ln} class must be integer-like; got {cf}")
        return None
    return int(rounded)


def main():
    global errors, warnings
    for split in SPLITS:
        for p in (ROOT / split).rglob("*.txt"):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            # treat whitespace-only as empty (valid negative)
            if not txt.strip():
                continue

            for ln, line in enumerate(txt.splitlines(), start=1):
                s = line.strip()
                if not s:
                    # blank line inside a label file—harmless, but flag once
                    warn(f"[FORMAT] {p}:{ln} empty line inside label file")
                    continue
                parts = s.split()
                if len(parts) != 5:
                    err(
                        f"[FIELDS] {p}:{ln} expected 5 fields, got {len(parts)} -> {s!r}"
                    )
                    continue

                cls = parse_class(parts[0], p, ln)
                if cls is None:
                    continue
                if cls not in ALLOWED:
                    err(
                        f"[CLASS] {p}:{ln} invalid class {cls}; allowed {sorted(ALLOWED)}"
                    )

                try:
                    x, y, w, h = map(float, parts[1:])
                except Exception as e:
                    err(f"[PARSE] {p}:{ln} box parse failed -> {s!r} ({e})")
                    continue

                if not is_finite(x, y, w, h):
                    err(
                        f"[FINITE] {p}:{ln} non-finite value in (x,y,w,h)=({x},{y},{w},{h})"
                    )
                    continue

                # Center coords within [0,1] (with epsilon tolerance)
                ok_xy = True
                ok_xy &= check_range("x", x, 0.0, 1.0, p, ln)
                ok_xy &= check_range("y", y, 0.0, 1.0, p, ln)

                # Width/height strictly > 0, and <= 1 (with epsilon tolerance)
                ok_wh = True
                ok_wh &= check_pos("w", w, 0.0, 1.0, p, ln)
                ok_wh &= check_pos("h", h, 0.0, 1.0, p, ln)

                # Corner check: ensure bbox fully inside [0,1] (with epsilon tolerance)
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                ok_corners = True
                ok_corners &= check_range("x1", x1, 0.0, 1.0, p, ln)
                ok_corners &= check_range("y1", y1, 0.0, 1.0, p, ln)
                ok_corners &= check_range("x2", x2, 0.0, 1.0, p, ln)
                ok_corners &= check_range("y2", y2, 0.0, 1.0, p, ln)

                # If corners fail hard but centers & sizes passed, call it out explicitly
                if (ok_xy and ok_wh) and not ok_corners:
                    warn(
                        f"[CORNERS] {p}:{ln} corners slightly out; consider clipping during export"
                    )

    if errors == 0:
        print(f"\nOK — no errors. {warnings} warning(s).")
        sys.exit(0)
    else:
        print(f"\nFound {errors} error(s), {warnings} warning(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
