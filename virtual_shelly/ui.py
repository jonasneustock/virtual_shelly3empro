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
    .chart { width: 100%; height: 260px; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 14px; }
    th { background: #fafafa; font-weight: 600; }
    .muted { color: #666; font-size: 12px; }
    code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
  </style>
  <script>
    let powerData = { history: [], forecast: [] };

    async function fetchOverview() {
      try {
        const res = await fetch('/admin/overview');
        const data = await res.json();
        render(data);
      } catch (e) {
        console.error('Fetch error', e);
      }
    }

    async function fetchPower() {
      try {
        const res = await fetch('/ui/power');
        const data = await res.json();
        powerData = { history: data.history || [], forecast: data.forecast || [] };
        document.getElementById('current-power').textContent = fmt(data.current);
        const ts = data.ts || (Date.now() / 1000);
        document.getElementById('power-updated').textContent = new Date(ts * 1000).toLocaleTimeString();
        updateModelMetrics(data.model || {});
        setForecastHorizon(data);
        renderForecastValues(powerData.forecast);
        drawPowerChart(powerData.history, powerData.forecast);
      } catch (e) {
        console.error('Power fetch error', e);
      }
    }

    function fmt(n, digits=2) {
      if (n === null || n === undefined) return '-';
      if (typeof n === 'number') return n.toFixed(digits);
      return String(n);
    }

    function updateModelMetrics(model) {
      const mse = model.mse;
      const mape = model.mape;
      document.getElementById('power-mse').textContent = mse === null || mse === undefined ? '-' : fmt(mse, 3);
      document.getElementById('power-mape').textContent = mape === null || mape === undefined ? '-' : fmt(mape, 2) + ' %';
      document.getElementById('power-samples').textContent = model.n || 0;
      const trainedAt = model.trained_at ? new Date(model.trained_at * 1000).toLocaleTimeString() : '-';
      document.getElementById('power-trained').textContent = trainedAt;
      document.getElementById('power-trained-n').textContent = model.n || 0;
      const total = model.dataset_total || 0;
      const btn = document.getElementById('train-btn');
      if (btn) {
        btn.textContent = `Train on dataset (${total} pts)`;
        btn.disabled = total < 2;
      }
    }

    function setForecastHorizon(data) {
      const horizon = (data.model && data.model.horizon) || (data.forecast ? data.forecast.length : 0);
      const el = document.getElementById('forecast-horizon');
      if (el) el.textContent = horizon || '-';
    }

    function renderForecastValues(forecast) {
      const el = document.getElementById('forecast-values');
      if (!el) return;
      if (!forecast || !forecast.length) {
        el.textContent = 'No forecast available';
        return;
      }
      const values = forecast.map(p => fmt(p.w, 1));
      const preview = values.slice(0, 10).join(', ');
      const suffix = values.length > 10 ? ' …' : '';
      el.textContent = `Next ${values.length}s: ${preview}${suffix}`;
    }

    async function triggerTraining() {
      const btn = document.getElementById('train-btn');
      if (btn) btn.disabled = true;
      try {
        await fetch('/ui/train', { method: 'POST' });
        await fetchPower();
      } catch (e) {
        console.error('Train error', e);
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    function drawPowerChart(history, forecast) {
      const canvas = document.getElementById('power-chart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const margin = 36;
      const plotW = Math.max(10, canvas.width - margin * 2);
      const plotH = Math.max(10, canvas.height - margin * 2);
      const nowSec = history.length ? history[history.length - 1].ts : (Date.now() / 1000);
      const windowSec = 60;
      const h = (history || []).filter(p => p.ts >= nowSec - windowSec);
      const f = forecast || [];
      const combined = h.concat(f);
      if (!combined.length) {
        ctx.fillStyle = '#999';
        ctx.font = '12px sans-serif';
        ctx.fillText('No power data yet', margin, canvas.height / 2);
        return;
      }

      let minTs = Math.min(...combined.map(p => p.ts));
      let maxTs = Math.max(...combined.map(p => p.ts));
      if (maxTs - minTs < 1) { maxTs = minTs + 1; }
      let minW = Math.min(...combined.map(p => p.w));
      let maxW = Math.max(...combined.map(p => p.w));
      if (Math.abs(maxW - minW) < 0.1) {
        maxW = minW + 1;
        minW = minW - 1;
      }

      function toXY(pt) {
        const x = margin + ((pt.ts - minTs) / (maxTs - minTs)) * plotW;
        const y = canvas.height - margin - ((pt.w - minW) / (maxW - minW)) * plotH;
        return [x, y];
      }

      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin, margin);
      ctx.lineTo(margin, canvas.height - margin);
      ctx.lineTo(canvas.width - margin, canvas.height - margin);
      ctx.stroke();

      function drawSeries(points, color, dashed=false) {
        if (!points.length) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash(dashed ? [6, 4] : []);
        points.forEach((pt, idx) => {
          const [x, y] = toXY(pt);
          if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }

      drawSeries(h, '#2563eb', false);
      drawSeries(f, '#d92c20', true);

      if (h.length) {
        const [x, y] = toXY(h[h.length - 1]);
        ctx.fillStyle = '#2563eb';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (f.length) {
        const [x, y] = toXY(f[0]);
        ctx.fillStyle = '#d92c20';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      const rangeLabel = document.getElementById('power-range');
      if (rangeLabel) {
        rangeLabel.textContent = `${Math.round(minW)}–${Math.round(maxW)} W window (${Math.round(maxTs - minTs)}s span)`;
      }
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

    window.addEventListener('load', () => {
      fetchOverview();
      fetchPower();
      setInterval(fetchOverview, 5000);
      setInterval(fetchPower, 2000);
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        fetchPower(); // fetch latest power + forecast for every chart redraw
      }, 200);
    });
  </script>
</head>
<body>
  <h1>Virtual Shelly 3EM Pro — Status <span class="muted" id="device"></span></h1>
  <div class="muted">Updated: <span id="updated">-</span></div>

  <div class="card" style="margin-top:12px">
    <h2>Power (live)</h2>
    <div class="row" style="margin-bottom:6px">
      <div>Current total: <b id="current-power">-</b> W</div>
      <div class="muted">Forecast for next <span id="forecast-horizon">30</span>s is shown in red.</div>
    </div>
    <div class="muted" style="margin-bottom:6px">Power updated: <span id="power-updated">-</span> · <span id="power-range"></span></div>
    <div class="muted" style="margin-bottom:6px">Forecast values: <span id="forecast-values">-</span></div>
    <div class="row muted" style="margin-bottom:6px">
      <div>MSE: <b id="power-mse">-</b></div>
      <div>MAPE: <b id="power-mape">-</b></div>
      <div>Samples: <b id="power-samples">0</b></div>
      <div>Last train: <span id="power-trained">-</span> (n=<span id="power-trained-n">0</span>)</div>
      <button id="train-btn" type="button" onclick="triggerTraining()">Train on dataset (0 pts)</button>
    </div>
    <canvas id="power-chart" class="chart"></canvas>
  </div>

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
