import glob
tot=small=0
IMGSZ=640
TH=2/IMGSZ  # ~2 px at imgsz
for split in ['train','val']:
    for p in glob.glob(r'data/raw/labels/%s/**/*.txt' % split, recursive=True):
        for ln in open(p, 'r', encoding='utf-8'):
            ln=ln.strip()
            if not ln: continue
            parts=ln.split()
            if len(parts)!=5: continue
            try:
                _,x,y,w,h = int(float(parts[0])), *map(float, parts[1:])
            except: 
                continue
            tot += 1
            if w < TH or h < TH:
                small += 1
print(f'Total boxes={tot}, tiny(<2px @{IMGSZ})={small} ({small/max(tot,1):.1%})')
