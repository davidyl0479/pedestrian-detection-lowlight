import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_index(paths):
    s=set()
    for ip in paths:
        with open(ip, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln and (ln.lower().endswith('.png') or ln.lower().endswith('.jpg') or ln.lower().endswith('.jpeg')):
                    s.add(ln.replace('/', '\\'))
    return sorted(s)

def check_one(p):
    try:
        with Image.open(p) as im:
            im.verify()  # cheaper than im.load()
        return (p, None)
    except (UnidentifiedImageError, OSError) as e:
        return (p, str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists', nargs='*', default=['index/train_mixed.txt','index/val_internal.txt','index/test_all.txt'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='logs/bad_images.txt')
    args = ap.parse_args()

    imgs = read_index(args.lists)
    Path('logs').mkdir(parents=True, exist_ok=True)
    print(f'Checking {len(imgs)} images from indices: {args.lists}')

    bad=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(check_one, p) for p in imgs]
        for i,f in enumerate(as_completed(futs),1):
            if i%1000==0: print(f'.. {i}/{len(imgs)} done')
            p,err=f.result()
            if err: bad.append((p,err))

    Path(args.out).write_text('\n'.join(f'{p} | {e}' for p,e in bad), encoding='utf-8')
    print(f'Done. Bad images: {len(bad)}. Wrote {args.out}')
if __name__=='__main__':
    main()
