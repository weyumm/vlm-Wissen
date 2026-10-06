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

  /* ---------- chapters ---------- */

  var CHAPTERS = (window.VLM_CHAPTERS || {})[key] || [];
  var chapterHost = document.getElementById('side-chapters');
  var chapterSection = document.getElementById('chapter-section');
  var currentChapter = 0;

  function chapterNumFromHash() {
    var m = /[#&]c=(\d+)/.exec(location.hash || '');
    if (!m) return 0;
    var n = parseInt(m[1], 10) - 1;
    return (n >= 0 && n < CHAPTERS.length) ? n : 0;
  }

  function renderChapters() {
    if (!chapterHost) return;
    chapterHost.innerHTML = '';
    if (!CHAPTERS.length) { chapterSection.hidden = true; return; }
    CHAPTERS.forEach(function (c, i) {
      var a = document.createElement('a');
      a.className = 'side-chapter';
      a.href = '#c=' + (i + 1);
      var label = c.title || ('第 ' + (i + 1) + ' 章');
      a.textContent = (i + 1) + '. ' + label;
      if (c.code) {
        var tag = document.createElement('span');
        tag.className = 'code-badge';
        tag.title = '这一节包含代码实现';
        tag.textContent = '{ }';
        a.appendChild(tag);
      }
      if (i === currentChapter) a.setAttribute('aria-current', 'page');
      a.addEventListener('click', function (e) {
        e.preventDefault();
        goChapter(i);
      });
      chapterHost.appendChild(a);
    });
  }

  function chapterPath(i) {
    return 'chapters/' + key + '/' + CHAPTERS[i].file + '.md';
  }

  function renderChapterNav() {
    var host = document.getElementById('chapter-nav');
    if (!host) return;
    host.innerHTML = '';
    var prev = currentChapter > 0 ? currentChapter - 1 : null;
    var next = currentChapter < CHAPTERS.length - 1 ? currentChapter + 1 : null;
    if (prev === null && next === null) { host.hidden = true; return; }
    host.hidden = false;
    if (prev !== null) {
      var p = document.createElement('a');
      p.className = 'chapter-btn';
      p.href = '#c=' + (prev + 1);
      p.textContent = '← ' + (CHAPTERS[prev].title || '上一章');
      p.addEventListener('click', function (e) { e.preventDefault(); goChapter(prev); });
      host.appendChild(p);
    }
    if (next !== null) {
      var n = document.createElement('a');
      n.className = 'chapter-btn next';
      n.href = '#c=' + (next + 1);
      n.textContent = (CHAPTERS[next].title || '下一章') + ' →';
      n.addEventListener('click', function (e) { e.preventDefault(); goChapter(next); });
      host.appendChild(n);
    }
  }

  function goChapter(i) {
    if (i === currentChapter) return;
    currentChapter = i;
    if (history.replaceState) {
      history.replaceState(null, '', '#c=' + (i + 1));
    } else {
      location.hash = 'c=' + (i + 1);
    }
    renderChapters();
    renderChapterNav();
    if (searchInput && searchInput.value) { searchInput.value = ''; }
    window.scrollTo(0, 0);
    loadChapter();
  }

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

  /* Formulas are rendered LAZILY: the placeholder is swapped for a small span
     holding the raw TeX at insert time, and KaTeX only runs when the span
     approaches the viewport. A chapter with a thousand formulas therefore
     paints instantly and only pays for the ones actually read. */
  var mathObserver = null;

  function renderPending(el) {
    var span = mathSpans[parseInt(el.getAttribute('data-mi'), 10)];
    if (!span) { el.remove(); return; }
    el.innerHTML = katex.renderToString(span.tex, {
      displayMode: span.display,
      throwOnError: false,
      strict: false
    });
    el.classList.remove('vlm-math-pending');
    el.removeAttribute('data-mi');
  }

  function hydrateMath() {
    if (mathObserver) { mathObserver.disconnect(); mathObserver = null; }
    var walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT, null);
    var targets = [];
    while (walker.nextNode()) {
      if (walker.currentNode.nodeValue.indexOf('@@VLMATH') >= 0) targets.push(walker.currentNode);
    }
    var pending = [];

    for (var t = 0; t < targets.length; t++) {
      var node = targets[t];
      var frag = document.createDocumentFragment();
      var value = node.nodeValue;
      var last = 0;
      var m;
      MATH_TOKEN.lastIndex = 0;
      while ((m = MATH_TOKEN.exec(value)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(value.slice(last, m.index)));
        var idx = parseInt(m[1], 10);
        var span = mathSpans[idx];
        if (span) {
          var ph = document.createElement('span');
          ph.className = 'vlm-math-pending' + (span.display ? ' display' : '');
          ph.setAttribute('data-mi', String(idx));
          ph.textContent = span.tex;
          frag.appendChild(ph);
          pending.push(ph);
        }
        last = m.index + m[0].length;
      }
      if (last < value.length) frag.appendChild(document.createTextNode(value.slice(last)));
      if (node.parentNode) node.parentNode.replaceChild(frag, node);
    }

    if (!pending.length) return;
    if (!('IntersectionObserver' in window)) {
      pending.forEach(renderPending);
      return;
    }
    mathObserver = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (!entries[i].isIntersecting) continue;
        mathObserver.unobserve(entries[i].target);
        renderPending(entries[i].target);
      }
    }, { rootMargin: '400px 0px' });
    pending.forEach(function (el) { mathObserver.observe(el); });

    /* Safety net: drain whatever the observer has not reached yet in small
       idle batches, so a very fast scroll or a Ctrl+F still ends up with
       every formula typeset — without ever blocking the first paint. */
    var queue = pending.slice();
    var drain = function () {
      if (!queue.length) return;
      var stop = Math.min(queue.length, 50);
      for (var k = 0; k < stop; k++) {
        var el = queue[k];
        if (el.isConnected && el.classList.contains('vlm-math-pending')) renderPending(el);
      }
      queue.splice(0, stop);
      if (queue.length) {
        if (window.requestIdleCallback) requestIdleCallback(drain, { timeout: 2000 });
        else setTimeout(drain, 250);
      }
    };
    setTimeout(drain, 1500);
  }

  /* ---------- images ---------- */

  /* Feishu callouts survived in the Markdown as blockquotes whose text now
     begins with the original emoji. Give them a card treatment. */
  var EMOJI_RE = /^\s*(\p{Extended_Pictographic}(?:\uFE0F|\u200D\p{Extended_Pictographic})*)/u;

  function enhanceCallouts() {
    var quotes = article.querySelectorAll('blockquote');
    for (var i = 0; i < quotes.length; i++) {
      var q = quotes[i];
      if (q.classList.contains('vlm-callout')) continue;
      /* The emoji may sit a few text nodes in (blockquote > p > text), and the
         first text node can be just a newline, so scan rather than only
         looking at the very first node. */
      var walker = document.createTreeWalker(q, NodeFilter.SHOW_TEXT, null);
      var node, m = null;
      while ((node = walker.nextNode())) {
        m = EMOJI_RE.exec(node.nodeValue);
        if (m) break;
        if (node.nodeValue.trim()) break;  // real content, no emoji -> give up
      }
      if (!m) continue;
      q.classList.add('vlm-callout');
      var range = document.createRange();
      range.setStart(node, m.index);
      range.setEnd(node, m.index + m[1].length);
      var badge = document.createElement('span');
      badge.className = 'callout-emoji';
      try { range.surroundContents(badge); } catch (e) { /* already styled */ }
    }
  }

  /* Wrap every <pre> in a frame with a language label and a copy button. */
  function enhanceCodeBlocks() {
    var pres = article.querySelectorAll('pre');
    for (var i = 0; i < pres.length; i++) {
      var pre = pres[i];
      if (!pre.parentNode) continue;
      if (pre.parentNode.classList && pre.parentNode.classList.contains('code-block')) continue;
      var code = pre.querySelector('code');
      var lang = '';
      if (code && code.className) {
        var lm = /language-([\w+#-]+)/.exec(code.className);
        if (lm) lang = lm[1];
      }
      var wrap = document.createElement('div');
      wrap.className = 'code-block';
      var head = document.createElement('div');
      head.className = 'code-head';
      var tag = document.createElement('span');
      tag.className = 'code-lang';
      tag.textContent = lang || 'code';
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.type = 'button';
      btn.textContent = '复制';
      btn.addEventListener('click', function () {
        var b = this;
        var c = b.parentNode.parentNode.querySelector('code');
        var text = c ? c.textContent : '';
        var done = function () {
          b.textContent = '已复制';
          b.classList.add('copied');
          setTimeout(function () { b.textContent = '复制'; b.classList.remove('copied'); }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {});
        } else {
          var ta = document.createElement('textarea');
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); done(); } catch (e) {}
          ta.remove();
        }
      });
      head.appendChild(tag);
      head.appendChild(btn);
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(head);
      wrap.appendChild(pre);
    }
  }

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

  var sourceUrl = function () {
    return CHAPTERS.length ? chapterPath(currentChapter) : part.md;
  };

  async function loadChapter() {
    if (!part) {
      article.innerHTML = '<div class="error">页面配置缺失：找不到 data-part 对应的分册。</div>';
      if (status) status.remove();
      return;
    }
    var url = sourceUrl();
    try {
      var res = await fetch(encodeURI(url));
      if (!res.ok) throw new Error('Markdown 加载失败（HTTP ' + res.status + '）');
      var md = await res.text();
      /* Chapter files reference ../../images/ so they resolve on GitHub; the
         page lives at the site root, so rewrite them back. */
      md = md.replace(/\.\.\/\.\.\/images\//g, 'images/');
      /* The leading nav line is GitHub-only navigation; drop it from the page. */
      md = md.replace(/^\[Previous\][^\n]*\n/, '');
      md = md.replace(/\n---\n\n\[Previous\][^\n]*\s*$/, '');
      /* "\$$" occurs once in the source as an escaped delimiter artifact. */
      md = md.replace(/\\\$\$/g, '$$$$');
      var prepared = extractMath(md);
      if (!window.marked || !window.DOMPurify || !window.katex) {
        throw new Error('渲染组件未载入，请确认 assets/vendor 下的脚本可以访问。');
      }
      marked.setOptions({ gfm: true, breaks: false });
      article.innerHTML = DOMPurify.sanitize(marked.parse(prepared), { ADD_ATTR: ['style', 'target'] });
      /* The document is on screen now; formulas fill in as they scroll. */
      if (status) status.remove();
      buildToc();
      observe();
      if (window.VLMTheme) window.VLMTheme.index(article);
      tuneImages();
      enhanceCallouts();
      enhanceCodeBlocks();
      Array.prototype.forEach.call(article.querySelectorAll('a[href^="http"]'), function (a) {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
      renderChapterNav();
      hydrateMath();
      prefetchNext();
    } catch (err) {
      article.innerHTML = '<div class="error">' + err.message +
        '<p>如果是双击打开的本文件，浏览器会阻止读取 Markdown。请在仓库根目录执行 <code>python -m http.server 8000</code>，再访问 <code>http://localhost:8000</code>。</p>' +
        '<p><a href="' + encodeURI(url) + '">打开本章 Markdown 原文</a></p></div>';
      if (status) status.remove();
    }
  }

  /* Warm the browser cache for the next chapter so flipping forward is instant. */
  function prefetchNext() {
    var next = currentChapter + 1;
    if (next >= CHAPTERS.length || !window.fetch) return;
    setTimeout(function () { fetch(encodeURI(chapterPath(next))).catch(function () {}); }, 2500);
  }

  function load() {
    currentChapter = chapterNumFromHash();
    renderChapters();
    renderChapterNav();
    loadChapter();
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

  if (article) {
    article.addEventListener('click', function (e) {
      if (e.target.tagName === 'IMG' && !e.target.classList.contains('img-missing')) openLightbox(e.target);
    });
  }

  var menu = document.getElementById('menu');
  if (menu) menu.addEventListener('click', function () { sidebar.classList.toggle('open'); });
  if (toc) toc.addEventListener('click', function (e) {
    if (e.target.closest('a')) sidebar.classList.remove('open');
  });
  if (chapterHost) chapterHost.addEventListener('click', function (e) {
    if (e.target.closest('a')) sidebar.classList.remove('open');
  });

  addEventListener('hashchange', function () {
    var i = chapterNumFromHash();
    if (i !== currentChapter) goChapter(i);
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
