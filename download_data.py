"""
Download and verify the Indian Pines hyperspectral dataset.

    python download_data.py                  # saves into ./data
    python download_data.py --out /path/dir  # choose another folder
    python download_data.py --verify-only    # just check files that are already there

Uses only the standard library + scipy/numpy (no requests, no Colab).
Each file is downloaded to a temporary name, then checked by loading it:
  - corrected cube  -> 3-D array, 145 x 145 x 200
  - ground truth    -> 2-D array, 145 x 145, labels 0..16, 10249 labelled pixels
Only files that pass are kept.
"""

import argparse
import os
import sys
import time
import urllib.request

import numpy as np
import scipy.io as sio

# name -> list of URLs tried in order (first one that works and passes the check wins)
FILES = {
    "Indian_pines_corrected.mat": [
        "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    ],
    "Indian_pines_gt.mat": [
        "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    ],
}

MANUAL_HELP = """
Automatic download failed. Download the two files by hand and put them in: {out}

  Indian_pines_corrected.mat   (about 5.7 MB)
  Indian_pines_gt.mat          (about 1 KB)

Places that host them (search for the file names):
  - University of the Basque Country page "Hyperspectral Remote Sensing Scenes"
  - IEEE DataPort: "Hyperspectral Remote Sensing Datasets: Indian Pines, Pavia University,
    Botswana and Salinas" (free account needed)

If the files are on your laptop, copy them to this machine, e.g.:
  scp Indian_pines_corrected.mat Indian_pines_gt.mat user@narnia:{out}/
Then run:  python download_data.py --verify-only --out {out}
"""


def find_array(path, ndim):
    m = sio.loadmat(path)
    for k, v in m.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == ndim:
            return k, v
    raise ValueError(f"no {ndim}-D array inside {os.path.basename(path)}")


def verify(path):
    """Return (ok, message). Loads the file and checks shape/content."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        return False, "missing"
    try:
        if "gt" in name.lower():
            key, a = find_array(path, 2)
            if a.shape != (145, 145):
                return False, f"unexpected shape {a.shape} (expected 145x145)"
            labelled = int((a > 0).sum())
            if a.max() != 16 or labelled != 10249:
                return False, f"labels look wrong (max {a.max()}, labelled pixels {labelled}; expected 16 / 10249)"
            return True, f"ok  key '{key}', shape {a.shape}, {labelled} labelled pixels, 16 classes"
        key, a = find_array(path, 3)
        if a.shape[:2] != (145, 145) or a.shape[2] not in (200, 220):
            return False, f"unexpected shape {a.shape} (expected 145x145x200)"
        return True, f"ok  key '{key}', shape {a.shape}, values {a.min()}..{a.max()}"
    except Exception as e:  # corrupted file, HTML error page saved as .mat, ...
        return False, f"cannot be read as a .mat file ({type(e).__name__}: {e})"


def download(url, dest, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (indian-pines-downloader)"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:5.2f} / {total / 1e6:.2f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data", help="folder to save the files in (default: ./data)")
    p.add_argument("--verify-only", action="store_true", help="do not download, only check files")
    p.add_argument("--retries", type=int, default=2, help="attempts per URL")
    args = p.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    all_ok = True

    for name, urls in FILES.items():
        dest = os.path.join(out, name)
        ok, msg = verify(dest)
        if ok:
            print(f"[skip] {name}: already present, {msg}")
            continue
        if args.verify_only:
            print(f"[FAIL] {name}: {msg}")
            all_ok = False
            continue
        if os.path.exists(dest):
            print(f"[warn] {name}: existing file is bad ({msg}); downloading again")
            os.remove(dest)

        got = False
        for url in urls:
            for attempt in range(1, args.retries + 1):
                print(f"[get ] {name}  <-  {url}  (attempt {attempt})")
                try:
                    download(url, dest)
                except Exception as e:
                    print(f"    failed: {type(e).__name__}: {e}")
                    if os.path.exists(dest + ".part"):
                        os.remove(dest + ".part")
                    time.sleep(1.5)
                    continue
                ok, msg = verify(dest)
                if ok:
                    print(f"[ ok ] {name}: {msg}")
                    got = True
                    break
                print(f"    downloaded but rejected: {msg}")
                os.remove(dest)
                break  # a bad payload will not fix itself on retry; try the next URL
            if got:
                break
        if not got:
            all_ok = False
            print(f"[FAIL] {name}: could not download a valid copy")

    if all_ok:
        print(f"\nAll good. Files are in: {out}")
        print("Next:\n  python indian_pines.py explore "
              f"--data '{os.path.join(out, 'Indian_pines_corrected.mat')}' "
              f"--gt '{os.path.join(out, 'Indian_pines_gt.mat')}'")
        return 0
    print(MANUAL_HELP.format(out=out))
    return 1


if __name__ == "__main__":
    sys.exit(main())