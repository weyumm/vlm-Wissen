/* Reading page: fetches one Markdown file, renders it with marked + KaTeX,
   then builds the table of contents, search, lightbox and scroll tracking.
   All dependencies are vendored under assets/vendor — no network needed. */
(function () {
  'use strict';

  var parts = window.VLM_PARTS || [];
  var key = document.body.getAttribute('data-part');
  var part = null;
  for (var i = 0; i < parts.length; i++) if (parts[i].key === key) part = parts[i];

  var article = document.getElementById('article');
  var toc = document.getElementById('toc');
  var sidebar = document.getElementById('sidebar');
  var searchInput = document.getElementById('search');
  var status = document.getElementById('status');

  /* ---------- part navigation + hero ---------- */

  function renderNav(host) {
    if (!host) return;
    host.innerHTML = '';
    parts.forEach(function (p) {
      var a = document.createElement('a');
      a.href = p.page;
      a.textContent = host.id === 'side-parts' ? p.label : p.short;
      if (p.key === key) a.setAttribute('aria-current', 'page');
      host.appendChild(a);
    });
  }

  function renderHero() {
    if (!part) return;
    var hero = document.getElementById('hero');
    if (!hero) return;
    var pills = part.pills.concat(part.meta).map(function (t) {
      return '<span class="pill">' + t + '</span>';
    }).join('');
    hero.innerHTML =
      '<div class="eyebrow">' + part.eyebrow + '</div>' +
      '<h1>' + part.label + '</h1>' +
      '<p>' + part.desc + '</p>' +
      '<div class="hero-meta">' + pills + '</div>';
    document.title = part.label + ' · VLM Wissen';
    var mdLink = document.getElementById('md-link');
    if (mdLink) mdLink.setAttribute('href', encodeURI(part.md));
  }

  renderNav(document.getElementById('part-nav'));
  renderNav(document.getElementById('side-parts'));
  renderHero();

  /* ---------- markdown + math ---------- */

  var MATH_TOKEN = /@@VLMATH(\d+)@@/g;
  var mathSpans = [];

  function extractMath(md) {
    mathSpans = [];
    /* Skip fenced code blocks only — `$$` there is a shell PID, not math.
       Inline code is deliberately NOT excluded: formulas such as
       \text{``a [class noun]''} contain backtick runs that would otherwise
       split a `$$…$$` pair in half. No fenced block in this repo contains `$$`. */
    var segments = md.split(/(```[\s\S]*?```)/);
    var out = '';
    for (var i = 0; i < segments.length; i++) {
      var seg = segments[i];
      if (i % 2 === 1) { out += seg; continue; }
      out += seg.replace(/\$\$([\s\S]+?)\$\$/g, function (full, tex, offset) {
        var display = tex.indexOf('\n') >= 0;
        if (!display) {
          /* A formula alone on its own line is display math. */
          var lineStart = seg.lastIndexOf('\n', offset) + 1;
          var lineEnd = seg.indexOf('\n', offset + full.length);
          if (lineEnd === -1) lineEnd = seg.length;
          if (seg.slice(lineStart, offset).trim() === '' && seg.slice(offset + full.length, lineEnd).trim() === '') {
            display = true;
          }
        }
        mathSpans.push({ tex: tex, display: display });
        return '@@VLMATH' + (mathSpans.length - 1) + '@@';
      });
    }
    return out;
  }

  function katexNode(span) {
    var box = document.createElement('span');
    try {
      box.innerHTML = katex.renderToString(span.tex, {
        displayMode: span.display,
        throwOnError: false,
        strict: false
      });
    } catch (e) {
      box.textContent = span.tex;
      box.className = 'math-fallback';
    }
    return box;
  }

  /* Replace placeholder text nodes with rendered math in chunks so the tab
     stays responsive on documents with thousands of formulas. */
  function hydrateMath() {
    var walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT, null);
    var targets = [];
    while (walker.nextNode()) {
      if (walker.currentNode.nodeValue.indexOf('@@VLMATH') >= 0) targets.push(walker.currentNode);
    }
    var total = mathSpans.length;
    var rendered = 0;
    var idx = 0;

    function step() {
      var stop = Math.min(targets.length, idx + 60);
      while (idx < stop) {
        var node = targets[idx++];
        var frag = document.createDocumentFragment();
        var value = node.nodeValue;
        var last = 0;
        var m;
        MATH_TOKEN.lastIndex = 0;
        while ((m = MATH_TOKEN.exec(value)) !== null) {
          if (m.index > last) frag.appendChild(document.createTextNode(value.slice(last, m.index)));
          var span = mathSpans[parseInt(m[1], 10)];
          if (span) { frag.appendChild(katexNode(span)); rendered++; }
          last = m.index + m[0].length;
        }
        if (last < value.length) frag.appendChild(document.createTextNode(value.slice(last)));
        if (node.parentNode) node.parentNode.replaceChild(frag, node);
      }
      if (status) status.textContent = '正在渲染公式… ' + rendered + ' / ' + total;
      if (idx < targets.length) { setTimeout(step, 0); } else { finish(); }
    }

    function finish() {
      if (status) status.remove();
      buildToc();
      observe();
      if (window.VLMTheme) window.VLMTheme.index(article);
    }

    if (!targets.length) { finish(); return; }
    step();
  }

  /* ---------- images ---------- */

  function tuneImages() {
    var imgs = article.querySelectorAll('img');
    for (var i = 0; i < imgs.length; i++) {
      var img = imgs[i];
      /* The first figures are usually above the fold — load them eagerly. */
      img.loading = i < 4 ? 'eager' : 'lazy';
      img.decoding = 'async';
      if (!img.getAttribute('alt')) img.setAttribute('alt', '讲义配图');
      img.addEventListener('error', function () { this.classList.add('img-missing'); });
    }
  }

  /* ---------- table of contents ---------- */

  function slug(text) {
    return text.toLowerCase()
      .replace(/<[^>]+>/g, '')
      .replace(/&[a-z0-9#]+;/gi, ' ')
      .replace(/[^\p{L}\p{N}]+/gu, '-')
      .replace(/^-|-$/g, '') || 'section';
  }

  function buildToc() {
    var heads = article.querySelectorAll('h1,h2,h3');
    var counts = Object.create(null);
    toc.innerHTML = '';
    var n = 0;
    for (var i = 0; i < heads.length; i++) {
      var h = heads[i];
      var text = h.textContent.trim();
      if (!text) continue;
      var id = slug(text);
      var seen = counts[id] || 0;
      counts[id] = seen + 1;
      if (seen) id = id + '-' + (seen + 1);
      h.id = id;
      var a = document.createElement('a');
      a.className = 'side-link depth-' + h.tagName.slice(1);
      a.href = '#' + id;
      a.textContent = text;
      toc.appendChild(a);
      n++;
    }
    if (!n) toc.innerHTML = '<span class="side-link">没有找到章节标题</span>';
  }

  function observe() {
    var links = toc.querySelectorAll('a');
    var targets = [];
    for (var i = 0; i < links.length; i++) {
      var el = document.getElementById(decodeURIComponent(links[i].hash.slice(1)));
      if (el) targets.push(el);
    }
    if (!targets.length || !('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      for (var j = 0; j < entries.length; j++) {
        if (!entries[j].isIntersecting) continue;
        var id = entries[j].target.id;
        for (var k = 0; k < links.length; k++) {
          links[k].classList.toggle('active', decodeURIComponent(links[k].hash.slice(1)) === id);
        }
      }
    }, { rootMargin: '-12% 0px -78% 0px' });
    targets.forEach(function (t) { io.observe(t); });
  }

  /* ---------- search ---------- */

  var BLOCKS = 'p,li,td,th,h1,h2,h3,h4,blockquote,pre';
  var marks = [];
  var hiddenEls = [];
  var bar = null;
  var searchTimer = null;
  var cursor = 0;

  function clearSearch() {
    for (var i = 0; i < marks.length; i++) {
      var mark = marks[i];
      if (mark.parentNode) mark.parentNode.replaceChild(document.createTextNode(mark.textContent), mark);
    }
    marks = [];
    cursor = 0;
    for (var j = 0; j < hiddenEls.length; j++) hiddenEls[j].style.display = '';
    hiddenEls = [];
    if (bar) { bar.remove(); bar = null; }
  }

  function resetTocFilter() {
    var links = toc.querySelectorAll('.side-link');
    for (var i = 0; i < links.length; i++) links[i].style.display = '';
  }

  function runSearch(raw) {
    clearSearch();
    resetTocFilter();
    var q = raw.trim().toLowerCase();
    if (q.length < 2) return;

    var links = toc.querySelectorAll('.side-link');
    for (var i = 0; i < links.length; i++) {
      if (links[i].textContent.toLowerCase().indexOf(q) < 0) links[i].style.display = 'none';
    }

    var walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
        var p = node.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        if (p.tagName === 'SCRIPT' || p.tagName === 'STYLE') return NodeFilter.FILTER_REJECT;
        if (p.closest('mark')) return NodeFilter.FILTER_REJECT;
        return node.nodeValue.toLowerCase().indexOf(q) >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });

    var found = [];
    var count = 0;
    while (walker.nextNode()) {
      var node = walker.currentNode;
      var value = node.nodeValue.toLowerCase();
      var perNode = [];
      var from = 0;
      var pos;
      while ((pos = value.indexOf(q, from)) !== -1) {
        perNode.push({ node: node, start: pos, end: pos + q.length });
        count++;
        from = pos + q.length;
        if (count >= 400) break;
      }
      /* Wrap backwards so earlier offsets on the same node stay valid. */
      perNode.reverse();
      found = found.concat(perNode);
      if (count >= 400) break;
    }

    for (var f = 0; f < found.length; f++) {
      var hit = found[f];
      var range = document.createRange();
      try {
        range.setStart(hit.node, hit.start);
        range.setEnd(hit.node, hit.end);
        var mark = document.createElement('mark');
        mark.className = 'vlm-hit';
        range.surroundContents(mark);
        marks.push(mark);
      } catch (e) { /* node changed shape; skip */ }
    }

    /* Keep every block holding a hit, plus its ancestors for context. */
    var keep = new Set();
    for (var k = 0; k < marks.length; k++) {
      var block = marks[k].parentElement ? marks[k].parentElement.closest(BLOCKS) : null;
      if (!block) continue;
      var up = block;
      while (up && up !== article) { keep.add(up); up = up.parentElement; }
    }

    var all = article.querySelectorAll(BLOCKS);
    for (k = 0; k < all.length; k++) {
      if (!keep.has(all[k])) { all[k].style.display = 'none'; hiddenEls.push(all[k]); }
    }

    bar = document.createElement('div');
    bar.className = 'search-bar';
    if (!marks.length) {
      bar.textContent = '没有匹配内容，试试更短的关键词。';
    } else {
      bar.textContent = '找到 ' + count + (count >= 400 ? ' 处以上' : ' 处') + '匹配；按 Enter 跳到下一处，Esc 清除。';
    }
    article.insertBefore(bar, article.firstChild);
    if (marks.length) marks[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function scheduleSearch() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () { runSearch(searchInput.value); }, 220);
  }

  /* ---------- lightbox ---------- */

  function openLightbox(img) {
    var box = document.createElement('div');
    box.className = 'lightbox';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', '放大图片');
    var big = document.createElement('img');
    big.src = img.currentSrc || img.src;
    big.alt = img.alt || '';
    var close = document.createElement('button');
    close.type = 'button';
    close.setAttribute('aria-label', '关闭');
    close.textContent = '\u00d7';
    box.appendChild(big);
    box.appendChild(close);
    document.body.appendChild(box);
    document.body.style.overflow = 'hidden';
    function dismiss() {
      box.remove();
      document.body.style.overflow = '';
      document.removeEventListener('keydown', onKey);
    }
    function onKey(e) { if (e.key === 'Escape') dismiss(); }
    close.addEventListener('click', dismiss);
    box.addEventListener('click', function (e) { if (e.target === box) dismiss(); });
    document.addEventListener('keydown', onKey);
    close.focus();
  }

  /* ---------- loading ---------- */

  async function load() {
    if (!part) {
      article.innerHTML = '<div class="error">页面配置缺失：找不到 data-part 对应的分册。</div>';
      if (status) status.remove();
      return;
    }
    try {
      var res = await fetch(encodeURI(part.md));
      if (!res.ok) throw new Error('Markdown 加载失败（HTTP ' + res.status + '）');
      var md = await res.text();
      /* "\$$" occurs once in the source as an escaped delimiter artifact. */
      md = md.replace(/\\\$\$/g, '$$$$');
      var prepared = extractMath(md);
      if (!window.marked || !window.DOMPurify || !window.katex) {
        throw new Error('渲染组件未载入，请确认 assets/vendor 下的脚本可以访问。');
      }
      marked.setOptions({ gfm: true, breaks: false });
      article.innerHTML = DOMPurify.sanitize(marked.parse(prepared), { ADD_ATTR: ['style', 'target'] });
      tuneImages();
      Array.prototype.forEach.call(article.querySelectorAll('a[href^="http"]'), function (a) {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
      article.addEventListener('click', function (e) {
        if (e.target.tagName === 'IMG' && !e.target.classList.contains('img-missing')) openLightbox(e.target);
      });
      hydrateMath();
    } catch (err) {
      article.innerHTML = '<div class="error">' + err.message +
        '<p>如果是双击打开的本文件，浏览器会阻止读取 Markdown。请在仓库根目录执行 <code>python -m http.server 8000</code>，再访问 <code>http://localhost:8000</code>。</p>' +
        '<p><a href="' + encodeURI(part.md) + '">打开 Markdown 原文</a></p></div>';
      if (status) status.remove();
    }
  }

  /* ---------- wiring ---------- */

  if (searchInput) {
    searchInput.addEventListener('input', scheduleSearch);
    searchInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        if (marks.length) {
          marks[cursor % marks.length].scrollIntoView({ behavior: 'smooth', block: 'center' });
          cursor++;
        }
      }
      if (e.key === 'Escape') {
        searchInput.value = '';
        runSearch('');
        searchInput.blur();
      }
    });
  }

  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (searchInput) searchInput.focus();
    }
  });

  var menu = document.getElementById('menu');
  if (menu) menu.addEventListener('click', function () { sidebar.classList.toggle('open'); });
  if (toc) toc.addEventListener('click', function (e) {
    if (e.target.closest('a')) sidebar.classList.remove('open');
  });

  var progress = document.getElementById('progress');
  if (progress) {
    addEventListener('scroll', function () {
      var max = document.documentElement.scrollHeight - innerHeight;
      progress.style.width = (max > 0 ? (scrollY / max) * 100 : 0) + '%';
    }, { passive: true });
  }

  load();
})();
