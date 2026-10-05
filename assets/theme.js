/* Theme handling shared by every page.
   Lecture text carries inline rgb() colors from the original export; those are
   remapped to dark-mode-safe equivalents so they stay readable on a dark canvas. */
(function () {
  var STORE_KEY = 'vlm-theme';

  /* normalized "rgb(r,g,b)" -> dark mode replacement */
  var DARK_MAP = {
    'rgb(36,91,219)': '#8fb0ff',
    'rgb(100,37,208)': '#c9a9ff',
    'rgb(216,57,49)': '#ff9b93',
    'rgb(46,161,33)': '#74dd68',
    'rgb(220,155,4)': '#ffc95c',
    'rgb(222,120,2)': '#ffab63',
    'rgb(255,165,61)': '#ffcd97',
    'rgb(98,210,86)': '#8fe07e',
    'rgb(247,105,100)': '#ff9f9a',
    'rgb(255,233,40)': '#ffef8a'
  };

  function norm(color) {
    return (color || '').replace(/\s+/g, '').toLowerCase();
  }

  var tinted = [];

  /* Collect elements that use one of the known lecture colors. */
  function collect(root) {
    tinted = [];
    if (!root) return;
    var els = root.querySelectorAll('[style]');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var c = el.style.color;
      if (!c) continue;
      var dark = DARK_MAP[norm(c)];
      if (!dark) continue;
      el.setAttribute('data-oc', c);
      tinted.push(el);
    }
  }

  function paint(isDark) {
    for (var i = 0; i < tinted.length; i++) {
      var el = tinted[i];
      if (isDark) {
        el.style.setProperty('color', DARK_MAP[norm(el.getAttribute('data-oc'))]);
      } else {
        el.style.setProperty('color', el.getAttribute('data-oc'));
      }
    }
  }

  function setDark(isDark) {
    document.body.classList.toggle('dark', isDark);
    paint(isDark);
    try { localStorage.setItem(STORE_KEY, isDark ? 'dark' : 'light'); } catch (e) { /* private mode */ }
  }

  window.VLMTheme = {
    isDark: function () { return document.body.classList.contains('dark'); },
    /* Call once after article content is in the DOM. */
    index: function (root) { collect(root); paint(document.body.classList.contains('dark')); },
    toggle: function () { setDark(!document.body.classList.contains('dark')); },
    init: function (button) {
      var saved = null;
      try { saved = localStorage.getItem(STORE_KEY); } catch (e) { /* ignore */ }
      if (saved === 'dark') document.body.classList.add('dark');
      if (button) button.addEventListener('click', function () { window.VLMTheme.toggle(); });
    }
  };

  /* Deferred scripts run before DOMContentLoaded, so this always fires. */
  document.addEventListener('DOMContentLoaded', function () {
    window.VLMTheme.init(document.getElementById('theme'));
  });
})();
