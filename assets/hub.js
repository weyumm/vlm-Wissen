/* Hub page: renders one card per part from the shared parts list. */
(function () {
  'use strict';
  var grid = document.getElementById('card-grid');
  if (!grid) return;
  var parts = window.VLM_PARTS || [];
  parts.forEach(function (p, i) {
    var idx = ('0' + (i + 1)).slice(-2);
    var a = document.createElement('a');
    a.className = 'card';
    a.href = p.page;
    var tags = p.pills.concat(p.meta).map(function (t) { return '<span>' + t + '</span>'; }).join('');
    a.innerHTML =
      '<div class="card-index">PART ' + idx + '</div>' +
      '<h2>' + p.label + '</h2>' +
      '<p>' + p.desc + '</p>' +
      '<div class="card-meta">' + tags + '</div>' +
      '<div class="card-go">进入阅读 →</div>';
    grid.appendChild(a);
  });
})();
