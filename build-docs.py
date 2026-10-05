# -*- coding: utf-8 -*-
"""Split the four source Markdown files into chapter files and refresh
assets/chapters.js so the reading pages know which files to load.

Source: 视觉多模态讲义（上）.md / 视觉多模态讲义（下）.md / 视觉多模态面试题.md / 视觉多模态项目.md
Output: chapters/<part>/NN-slug.md
        assets/chapters.js   (window.VLM_CHAPTERS = { partKey: [{file, title}] })
        chapters-index.md    (Markdown chapters list per part, used by the GitHub README)

Each chapter file starts and ends with a nav line of the form

    [Previous](NN-prev.md) | [Contents](../../README.md) | [Next](NN-next.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/<page>#c=<idx>)

Image references in chapter files are rewritten from images/ to ../../images/
so they resolve correctly when viewed on GitHub (chapters/ is two levels deep).
The site rewrites them back to images/ before rendering.
"""
import os, re, io, json

REPO = r"D:\同车行与大数据\多模态知识库\vlm-Wissen"

H1_RE = re.compile(r"^# (?!#)(.+?)$")
H2_RE = re.compile(r"^## (?!#)(.+?)$")

# A chapter larger than SPLIT_KB is divided along its H2 headings; a
# sub-section smaller than MERGE_KB is folded into its neighbour so we do not
# ship one-paragraph files.
SPLIT_KB = 110
MERGE_KB = 8

PARTS = [
    ("lecture-1",  "视觉多模态讲义（上）.md", "lecture-1",  "lecture-1.html"),
    ("lecture-2",  "视觉多模态讲义（下）.md", "lecture-2",  "lecture-2.html"),
    ("interview",  "视觉多模态面试题.md",   "interview",  "interview.html"),
    ("projects",   "视觉多模态项目.md",     "projects",   "projects.html"),
]


def headings_outside_code(text, rx):
    """(offset, title) pairs for headings matching rx, skipping fenced code."""
    lines = text.split("\n")
    out = []
    in_code = False
    pos = 0
    for ln in lines:
        if ln.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            m = rx.match(ln)
            if m:
                out.append((pos, m.group(1).strip()))
        pos += len(ln) + 1
    return out


def h1_positions_outside_code(text):
    """Char offsets of H1 lines that are NOT inside fenced code blocks."""
    return [p for p, _ in headings_outside_code(text, H1_RE)]


def split_h1(text):
    positions = h1_positions_outside_code(text)
    if not positions:
        yield ("", text)
        return
    if positions[0] > 0:
        yield ("", text[:positions[0]])
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        head = re.search(r"(?m)^# (.+?)$", text[start:end])
        title = head.group(1).strip() if head else ""
        yield (title, text[start:end])


def split_h2(body):
    """Split a chunk along its H2 headings; the leading piece (which holds the
    H1 line itself) is folded into the first H2 section."""
    hits = headings_outside_code(body, H2_RE)
    if not hits:
        return [body]
    pieces = []
    if hits[0][0] > 0:
        pieces.append(body[:hits[0][0]])
    for i, (start, _) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(body)
        pieces.append(body[start:end])
    # fold the H1-only lead-in into the first real section
    if len(pieces) > 1:
        pieces[1] = pieces[0] + pieces[1]
        pieces = pieces[1:]
    return pieces


def unescape(s):
    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")

def clean_title(raw):
    """Strip inline HTML and leading '1. ' numbering from a heading."""
    t = re.sub(r"<[^>]+>", "", raw).strip()
    t = unescape(t)
    t = re.sub(r"^\d+(\.\d+)*\s*\.?\s*", "", t).strip()
    return t

def slug_for(title, fallback):
    t = clean_title(title)
    if not t:
        return fallback
    s = re.sub(r"\s+", "-", t)
    s = re.sub(r"[^\w一-鿿-]+", "", s)
    return s.strip("-") or fallback


def rewrite_images(body):
    body = body.replace("(./images/", "(../../images/")
    body = body.replace("(images/", "(../../images/")
    body = body.replace("] (images/", "] (../../images/")
    return body


def build_nav_line(prev_slug, next_slug, page, idx):
    base = "https://weyumm.github.io/vlm-Wissen/"
    parts = []
    if prev_slug:
        parts.append(f"[Previous]({prev_slug}.md)")
    parts.append("[Contents](../../README.md)")
    if next_slug:
        parts.append(f"[Next]({next_slug}.md)")
    parts.append(f"[Visual website]({base}{page}#c={idx})")
    return " | ".join(parts)


def split_part(part_key, src_md, page):
    src_path = os.path.join(REPO, src_md)
    text = open(src_path, encoding="utf-8").read()
    sections = list(split_h1(text))
    chapters = []  # (slug, title, body)
    out_dir = os.path.join(REPO, "chapters", part_key)
    os.makedirs(out_dir, exist_ok=True)

    # Anything before the first H1 is merged into chapter 1 so the page never
    # opens on an almost-empty "preface" chapter.
    preamble = ""
    if sections and not sections[0][0]:
        preamble = sections[0][1].strip()
        sections = sections[1:]

    for i, (title, body) in enumerate(sections):
        if i == 0 and preamble:
            body = preamble + "\n\n" + body
        parent = clean_title(title)
        if len(body) <= SPLIT_KB * 1024:
            chunks = [(title, body)]
        else:
            pieces = split_h2(body)
            # (h2 raw title, text); the first piece carries the H1 line
            merged = []
            for j, piece in enumerate(pieces):
                hit = re.search(r"(?m)^## (?!#)(.+?)$", piece)
                sub = hit.group(1).strip() if hit else ""
                if merged and len(piece) < MERGE_KB * 1024:
                    merged[-1][1] += piece
                elif merged and len(merged[-1][1]) < MERGE_KB * 1024:
                    merged[-1][1] += piece
                    merged[-1][0] = sub or merged[-1][0]
                else:
                    merged.append([sub, piece])
            chunks = []
            for sub, piece in merged:
                label = parent + " · " + clean_title(sub) if sub else parent
                chunks.append((label, piece))
        for chunk_title, chunk_body in chunks:
            chapters.append([chunk_title, chunk_body])

    # Number slugs sequentially and keep them unique.
    used = set()
    for i, ch in enumerate(chapters):
        slug = f"{i + 1:02d}-{slug_for(ch[0], 'chapter')}"
        n = 2
        while slug in used:
            slug = f"{i + 1:02d}-{slug_for(ch[0], 'chapter')}-{n}"
            n += 1
        used.add(slug)
        ch.insert(0, slug)

    for i, (slug, title, body) in enumerate(chapters):
        prev_slug = chapters[i - 1][0] if i > 0 else None
        next_slug = chapters[i + 1][0] if i + 1 < len(chapters) else None
        nav_top = build_nav_line(prev_slug, next_slug, page, i + 1)
        nav_bottom = build_nav_line(prev_slug, next_slug, page, i + 1)
        out = nav_top + "\n\n" + rewrite_images(body).rstrip() + "\n\n---\n\n" + nav_bottom + "\n"
        path = os.path.join(out_dir, slug + ".md")
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"  wrote chapters/{part_key}/{slug}.md ({len(body)} bytes) — {clean_title(title)[:40]}")

    clean_titles = [clean_title(t) for _, t, _ in chapters]
    return [(c[0], t) for c, t in zip(chapters, clean_titles)]


def write_chapters_js(summary):
    """Write assets/chapters.js with window.VLM_CHAPTERS."""
    path = os.path.join(REPO, "assets", "chapters.js")
    buf = io.StringIO()
    buf.write("/* Generated by build-docs.py. Maps a part key to its chapter list. */\n")
    buf.write("window.VLM_CHAPTERS = {\n")
    for part_key, chapters in summary.items():
        buf.write(f"  {part_key!r}: [\n")
        for slug, title in chapters:
            t = title.replace("\\", "\\\\").replace("'", "\\'")
            buf.write(f"    {{ file: {slug!r}, title: '{t}' }},\n")
        buf.write("  ],\n")
    buf.write("};\n")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    print(f"  wrote {path}")


def write_chapters_index(summary):
    """Write chapters-index.md so the GitHub README can include a chapter list."""
    path = os.path.join(REPO, "chapters", "INDEX.md")
    parts_meta = {p[0]: (p[1].replace(".md", ""), p[3]) for p in PARTS}
    buf = io.StringIO()
    for part_key, chapters in summary.items():
        title, page = parts_meta[part_key]
        buf.write(f"## {title}\n\n")
        for slug, t in chapters:
            buf.write(f"- [`{slug}.md`]({part_key}/{slug}.md) — {t}\n")
        buf.write("\n")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    print(f"  wrote {path}")


def main():
    summary = {}
    for part_key, src_md, folder, page in PARTS:
        print(f"=== {part_key} ===")
        chapters = split_part(part_key, src_md, page)
        summary[part_key] = chapters
    write_chapters_js(summary)
    write_chapters_index(summary)


if __name__ == "__main__":
    main()