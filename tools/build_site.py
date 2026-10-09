#!/usr/bin/env python3
"""Build a static, portable reading site from the repository's Markdown."""
from html import escape
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs'
TAG = '2026-10-09-full-stack'
RAW = f'https://raw.githubusercontent.com/xangma/pycbc-jax-benchmarks/{TAG}/'
BLOB = f'https://github.com/xangma/pycbc-jax-benchmarks/blob/{TAG}/'


def page_path(source):
    relative = source.relative_to(ROOT)
    return relative.with_name('index.html') if relative.name == 'README.md' else relative.with_suffix('.html')


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir()
    pages = sorted(p for p in ROOT.rglob('*.md') if not any(
        part in {'.git', 'docs', '__pycache__', '.venv', 'inputs', 'sources', 'results', 'resolved'}
        for part in p.relative_to(ROOT).parts))
    for source in pages:
        target = DEST / page_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = markdown.markdown(source.read_text(), extensions=['tables', 'fenced_code'])
        def replace(match):
            attribute, url = match.groups()
            split = urlsplit(url)
            if split.scheme or split.netloc or not split.path:
                return match.group(0)
            local = (source.parent / unquote(split.path)).resolve()
            if not local.is_relative_to(ROOT) or not local.is_file():
                raise ValueError(f'Broken public link in {source.relative_to(ROOT)}: {url}')
            relative = local.relative_to(ROOT)
            if local.suffix == '.md':
                linked = page_path(local)
                link = __import__('os').path.relpath(DEST / linked, target.parent)
            elif local.suffix == '.png':
                copied = DEST / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, copied)
                link = __import__('os').path.relpath(copied, target.parent)
            else:
                link = (RAW if local.suffix in {'.json', '.csv', '.gz', '.rst'} else BLOB) + quote(relative.as_posix())
            if split.fragment:
                link += '#' + split.fragment
            return f'{attribute}="{escape(link, quote=True)}"'
        body = re.sub(r'(href|src)="([^"]+)"', replace, body)
        title_match = re.search(r'<h1>(.*?)</h1>', body)
        title = title_match.group(1) if title_match else 'PyCBC JAX benchmarks'
        target.write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{max-width:1100px;margin:2rem auto;padding:0 1.2rem;font:17px/1.6 system-ui,sans-serif;color:#172a3a;background:#fff}}
a{{color:#165b94}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:.45rem .8rem;border:1px solid #c6d2dc;text-align:left}}pre{{padding:1rem;overflow:auto;background:#eef3f7;border-radius:5px}}code{{font-size:.88em}}h1,h2,h3{{line-height:1.2}}nav{{font-size:.9rem;border-bottom:1px solid #c6d2dc;padding-bottom:1rem}}
</style></head><body><nav><a href="https://xangma.github.io/pycbc-jax-benchmarks/">Campaign index</a> · <a href="https://github.com/xangma/pycbc-jax-benchmarks/releases/tag/{TAG}">Frozen release</a> · <a href="{BLOB}{quote(source.relative_to(ROOT).as_posix())}">Source</a></nav>{body}</body></html>\n''')
    (DEST / '.nojekyll').write_text('')
    print(f'Built {len(pages)} pages; all local source links resolved.')


if __name__ == '__main__':
    main()
