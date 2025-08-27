from pathlib import Path
from PIL import Image, UnidentifiedImageError
bad=[]
for split in ('train','val'):
    for p in Path('data/raw/images', split).rglob('*.png'):
        try:
            with Image.open(p) as im:
                im.load()  # forces full decode; triggers CRC issues
        except (UnidentifiedImageError, OSError) as e:
            bad.append((str(p), str(e)))
open('logs/bad_images.txt','w',encoding='utf-8').write(
    '\n'.join(f'{p} | {e}' for p,e in bad)
)
print(f'Checked. Bad images: {len(bad)}. See logs/bad_images.txt')
