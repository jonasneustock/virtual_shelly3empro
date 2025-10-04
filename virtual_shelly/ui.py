from __future__ import annotations

def dashboard_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Virtual Shelly 3EM Pro — Status</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #0b0b0b; }
    h1 { font-size: 20px; margin: 0 0 12px 0; }
    h2 { font-size: 16px; margin: 18px 0 10px 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #e2e2e2; border-radius: 8px; padding: 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 14px; }
    th { background: #fafafa; font-weight: 600; }
    .muted { color: #666; font-size: 12px; }
    code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
    .charts { display: grid; grid-template-columns: 1fr; gap: 16px; }
    .chart-wrap { border: 1px solid #e2e2e2; border-radius: 8px; padding: 12px; }
    .legend { margin: 6px 0 0 0; font-size: 12px; }
    .legend span { display: inline-block; margin-right: 12px; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
  </style>
  <script>
    // Rolling buffers for graphs (last ~10 minutes at 5s interval = 120 points)
    const MAX_POINTS = 120;
    const hist = {
      ts: [], total: [], a: [], b: [], c: []
    };

    async function fetchOverview() {
      try {
        const res = await fetch('/admin/overview');
        const data = await res.json();
        render(data);
        updateHistoryAndDraw(data);
      } catch (e) {
        console.error('Fetch error', e);
      }
    }

    function fmt(n, digits=2) {
      if (n === null || n === undefined) return '-';
      if (typeof n === 'number') return n.toFixed(digits);
      return String(n);
    }

    function render(d) {
      const dev = d.device || {};
      document.getElementById('device').textContent = (dev.id || 'device') + ' (' + (dev.model || '') + ' ' + (dev.ver || '') + ')';
      document.getElementById('updated').textContent = new Date(d.ts * 1000).toLocaleString();

      const em = (d.values && d.values.em) || {};
      const emdata = (d.values && d.values.emdata) || {};

      const cur = [
        ['A Voltage (V)', fmt(em.a_voltage)],
        ['B Voltage (V)', fmt(em.b_voltage)],
        ['C Voltage (V)', fmt(em.c_voltage)],
        ['A Current (A)', fmt(em.a_current)],
        ['B Current (A)', fmt(em.b_current)],
        ['C Current (A)', fmt(em.c_current)],
        ['A Power (W)', fmt(em.a_act_power)],
        ['B Power (W)', fmt(em.b_act_power)],
        ['C Power (W)', fmt(em.c_act_power)],
        ['Power Factor A', fmt(em.a_pf, 3)],
        ['Power Factor B', fmt(em.b_pf, 3)],
        ['Power Factor C', fmt(em.c_pf, 3)],
        ['Frequency (Hz)', fmt(em.frequency, 2)],
        ['Total Power (W)', fmt(em.total_act_power)],
      ];
      document.getElementById('current-tbody').innerHTML = cur.map(([k,v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

      const energy = [
        ['A Import (kWh)', fmt(emdata.a_total_act_energy, 3)],
        ['B Import (kWh)', fmt(emdata.b_total_act_energy, 3)],
        ['C Import (kWh)', fmt(emdata.c_total_act_energy, 3)],
        ['A Export (kWh)', fmt(emdata.a_total_act_ret_energy, 3)],
        ['B Export (kWh)', fmt(emdata.b_total_act_ret_energy, 3)],
        ['C Export (kWh)', fmt(emdata.c_total_act_ret_energy, 3)],
        ['Total Import (kWh)', fmt(emdata.total_act, 3)],
        ['Total Export (kWh)', fmt(emdata.total_act_ret, 3)],
        ['Period (s)', fmt(emdata.period, 0)],
      ];
      document.getElementById('energy-tbody').innerHTML = energy.map(([k,v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

      // Metrics
      const http = (d.metrics && d.metrics.http) || {};
      const ws = (d.metrics && d.metrics.ws) || {};
      const udp = (d.metrics && d.metrics.udp) || {};
      document.getElementById('http-total').textContent = (http.total ?? 0);
      const bym = http.by_method || {};
      const rows = Object.keys(bym).sort().map(m => `<tr><td><code>${m}</code></td><td>${bym[m]}</td></tr>`).join('');
      document.getElementById('http-by-method').innerHTML = rows || '<tr><td colspan="2" class="muted">No calls yet</td></tr>';
      document.getElementById('ws-stats').textContent = `${ws.clients || 0} clients, ${ws.rpc_messages || 0} RPC msgs, ${ws.notify_total || 0} notifies`;
      document.getElementById('udp-stats').textContent = `${udp.packets || 0} packets, ${udp.replies || 0} replies`;

      // Clients
      function listToRows(arr, cols) {
        return (arr||[]).slice().reverse().map(x => `<tr>${cols.map(c => `<td>${x[c] ?? '-'}</td>`).join('')}</tr>`).join('');
      }
      document.getElementById('http-recent').innerHTML = listToRows(d.clients?.http_recent, ['ts','ip','verb','method']);
      document.getElementById('ws-recent').innerHTML = listToRows(d.clients?.ws_recent, ['ts','ip','event','method']);
      document.getElementById('udp-recent').innerHTML = listToRows(d.clients?.udp_recent, ['ts','ip','method']);
      const wsCon = (d.clients?.ws_connected || []).map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('ws-connected').innerHTML = wsCon || '<span class="muted">None</span>';

      // Unique clients
      const httpU = d.clients?.http_unique || [];
      const wsU = d.clients?.ws_unique || [];
      const udpU = d.clients?.udp_unique || [];
      document.getElementById('http-uniq-count').textContent = httpU.length;
      document.getElementById('ws-uniq-count').textContent = wsU.length;
      document.getElementById('udp-uniq-count').textContent = udpU.length;
      document.getElementById('http-uniq-list').innerHTML = httpU.map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('ws-uniq-list').innerHTML = wsU.map(ip => `<code>${ip}</code>`).join(', ');
      document.getElementById('udp-uniq-list').innerHTML = udpU.map(ip => `<code>${ip}</code>`).join(', ');
    }

    function pushRolling(arr, v) {
      arr.push(v);
      if (arr.length > MAX_POINTS) arr.shift();
    }

    function updateHistoryAndDraw(d) {
      const em = (d.values && d.values.em) || {};
      const total = Number(em.total_act_power ?? 0);
      const a = Number(em.a_act_power ?? 0);
      const b = Number(em.b_act_power ?? 0);
      const c = Number(em.c_act_power ?? 0);
      const ts = (d.ts || Math.floor(Date.now()/1000));
      pushRolling(hist.ts, ts);
      pushRolling(hist.total, total);
      pushRolling(hist.a, a);
      pushRolling(hist.b, b);
      pushRolling(hist.c, c);
      drawPowerChart();
    }

    function drawPowerChart() {
      const canvas = document.getElementById('power-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0,0,W,H);

      // Compute Y scale with padding
      const series = [hist.total, hist.a, hist.b, hist.c];
      let minV = Infinity, maxV = -Infinity;
      for (const s of series) {
        for (const v of s) { if (!Number.isFinite(v)) continue; minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
      }
      if (!Number.isFinite(minV) || !Number.isFinite(maxV)) { minV = 0; maxV = 1; }
      if (minV === maxV) { maxV = minV + 1; }
      const pad = (maxV - minV) * 0.1;
      minV -= pad; maxV += pad;

      // Grid
      ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
      const gridLines = 4;
      for (let i=0;i<=gridLines;i++) {
        const y = i*(H/gridLines);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
      }

      // Map function
      const n = hist.ts.length;
      const xFor = idx => (n <= 1) ? 0 : (idx/(n-1))*W;
      const yFor = v => H - ((v - minV)/(maxV - minV))*H;

      // Draw a polyline helper
      function drawLine(data, color) {
        if (data.length < 2) return;
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
        for (let i=0;i<data.length;i++) {
          const x = xFor(i); const y = yFor(Number(data[i]));
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke();
      }

      drawLine(hist.total, '#111');
      drawLine(hist.a, '#e53935');
      drawLine(hist.b, '#1e88e5');
      drawLine(hist.c, '#43a047');

      // Axes labels (min/max)
      ctx.fillStyle = '#666'; ctx.font = '12px sans-serif';
      ctx.fillText(maxV.toFixed(1)+' W', 4, 12);
      ctx.fillText(minV.toFixed(1)+' W', 4, H-4);
    }

    window.addEventListener('load', () => {
      fetchOverview();
      setInterval(fetchOverview, 5000);
    });
  </script>
</head>
<body>
  <h1>Virtual Shelly 3EM Pro — Status <span class="muted" id="device"></span></h1>
  <div class="muted">Updated: <span id="updated">-</span></div>

  <div class="grid">
    <div class="card">
      <h2>Current Values</h2>
      <table>
        <tbody id="current-tbody"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Energy Counters</h2>
      <table>
        <tbody id="energy-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>HTTP Metrics</h2>
      <div>Total: <b id="http-total">0</b></div>
      <table>
        <thead><tr><th>Method</th><th>Count</th></tr></thead>
        <tbody id="http-by-method"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>WS & UDP Metrics</h2>
      <div>WebSocket: <b id="ws-stats">-</b></div>
      <div>UDP: <b id="udp-stats">-</b></div>
      <div style="margin-top:8px">WS Connected: <span id="ws-connected"></span></div>
    </div>
  </div>

  <div class="charts">
    <div class="chart-wrap">
      <h2>Power — Total and Phases (W)</h2>
      <canvas id="power-canvas" width="900" height="240"></canvas>
      <div class="legend">
        <span><span class="dot" style="background:#111"></span>Total</span>
        <span><span class="dot" style="background:#e53935"></span>A</span>
        <span><span class="dot" style="background:#1e88e5"></span>B</span>
        <span><span class="dot" style="background:#43a047"></span>C</span>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Unique Clients</h2>
    <div>HTTP: <b id="http-uniq-count">0</b> <span id="http-uniq-list"></span></div>
    <div>WS: <b id="ws-uniq-count">0</b> <span id="ws-uniq-list"></span></div>
    <div>UDP RPC: <b id="udp-uniq-count">0</b> <span id="udp-uniq-list"></span></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Recent HTTP Clients</h2>
      <table>
        <thead><tr><th>TS</th><th>IP</th><th>Verb</th><th>Method</th></tr></thead>
        <tbody id="http-recent"></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Recent WS Clients</h2>
      <table>
        <thead><tr><th>TS</th><th>IP</th><th>Event</th><th>Method</th></tr></thead>
        <tbody id="ws-recent"></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Recent UDP Clients</h2>
    <table>
      <thead><tr><th>TS</th><th>IP</th><th>Method</th></tr></thead>
      <tbody id="udp-recent"></tbody>
    </table>
  </div>

</body>
</html>
"""
