const API = "";
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

// ---- Chat ----
function addMessage(text, role) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    addMessage(msg, "user");
    chatInput.value = "";
    sendBtn.disabled = true;

    const thinking = document.createElement("div");
    thinking.className = "msg agent loading";
    thinking.textContent = "Thinking...";
    chatMessages.appendChild(thinking);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: msg }),
        });
        const data = await res.json();
        thinking.remove();
        addMessage(data.reply || "No response.", "agent");
    } catch (err) {
        thinking.remove();
        addMessage("Error: " + err.message, "agent");
    }
    sendBtn.disabled = false;
    chatInput.focus();
}

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ---- Regime Card ----
async function loadRegime() {
    const el = document.getElementById("regimeContent");
    el.innerHTML = '<span class="loading">Fetching live regime...</span>';
    try {
        const res = await fetch(`${API}/api/regime`);
        const d = await res.json();
        if (d.status !== "success") {
            el.innerHTML = `<span class="empty-msg">${d.error_message || "Failed."}</span>`;
            return;
        }
        const m = d.metrics || {};
        el.innerHTML = `
            <div class="regime-badge ${d.regime}">${d.regime}</div>
            <div style="margin-bottom:8px;font-size:0.85rem;">Strategy: <strong>${d.strategy}</strong></div>
            <div class="metric-row"><span class="label">Nifty Close</span><span class="value">${m.close}</span></div>
            <div class="metric-row"><span class="label">50-DMA</span><span class="value">${m.dma_50}</span></div>
            <div class="metric-row"><span class="label">DMA Slope</span><span class="value">${m.dma_50_slope}</span></div>
            <div class="metric-row"><span class="label">20d Return</span><span class="value">${(m.return_20d * 100).toFixed(2)}%</span></div>
            <div class="metric-row"><span class="label">Volatility</span><span class="value">${(m.volatility * 100).toFixed(2)}%</span></div>
            <div style="margin-top:10px;font-size:0.78rem;color:var(--text-secondary);">
                Source: ${d.source} | ${d.fetched_at_ist}<br>
                Last trade: ${d.last_trade_date}
            </div>
        `;
    } catch (err) {
        el.innerHTML = `<span class="empty-msg">Error: ${err.message}</span>`;
    }
}

// ---- Nifty 50 Signal Board ----
function signalClass(signal) {
    if (signal === "BUY") return "signal-buy";
    if (signal === "SELL") return "signal-sell";
    return "signal-hold";
}

function formatInr(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function loadSignalBoard() {
    const el = document.getElementById("signalBoardContent");
    el.innerHTML = '<span class="loading">Scanning Nifty 50 signals...</span>';
    try {
        const res = await fetch(`${API}/api/signals/nifty50?include_news=true&max_news=2&news_days=1`);
        const d = await res.json();
        if (d.status !== "success") {
            el.innerHTML = `<span class="empty-msg">${d.error_message || "Failed to build signal board."}</span>`;
            return;
        }

        const counts = d.signal_counts || {};
        let html = `
            <div class="signal-summary">
                <div class="metric-row"><span class="label">Regime</span><span class="value">${d.regime || "-"}</span></div>
                <div class="metric-row"><span class="label">Strategy</span><span class="value">${d.strategy || "-"}</span></div>
                <div class="metric-row"><span class="label">Signals</span><span class="value">BUY ${counts.BUY || 0} | HOLD ${counts.HOLD || 0} | SELL ${counts.SELL || 0}</span></div>
                <div class="metric-row"><span class="label">Scanned</span><span class="value">${d.stocks_scanned || 0}/${d.stocks_requested || 0}</span></div>
            </div>
        `;

        const rows = Array.isArray(d.signals) ? d.signals : [];
        if (rows.length === 0) {
            html += '<div class="chart-empty" style="margin-top:10px;">No signal rows available.</div>';
        } else {
            html += `
                <div class="signal-table-wrap">
                    <table class="signal-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Signal</th>
                                <th>Price</th>
                                <th>Entry</th>
                                <th>Stop</th>
                                <th>Target</th>
                                <th>RSI</th>
                                <th>News (Today/1D)</th>
                                <th>Rationale</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            for (const r of rows) {
                const newsItems = Array.isArray(r.news) ? r.news : [];
                const newsSummary = newsItems.length > 0
                    ? newsItems.map((n) => `${n.title} (${n.publisher})`).join(" | ")
                    : (r.news_error ? `News unavailable: ${r.news_error}` : "No recent headlines");
                html += `
                    <tr>
                        <td>${r.display_symbol || r.symbol}</td>
                        <td><span class="signal-pill ${signalClass(r.signal)}">${r.signal || "HOLD"}</span></td>
                        <td>${formatInr(r.current_price)}</td>
                        <td>${formatInr(r.entry)}</td>
                        <td>${formatInr(r.stop)}</td>
                        <td>${formatInr(r.target)}</td>
                        <td>${r.metrics && r.metrics.rsi !== null && r.metrics.rsi !== undefined ? Number(r.metrics.rsi).toFixed(2) : "-"}</td>
                        <td class="signal-news">${r.news_today_count || 0}/${r.news_recent_count || 0} - ${newsSummary}</td>
                        <td class="signal-rationale">${r.rationale || "-"}</td>
                    </tr>
                `;
            }
            html += `
                        </tbody>
                    </table>
                </div>
            `;
        }

        if (d.scan_errors && d.scan_errors.length > 0) {
            html += `<div class="empty-msg" style="margin-top:10px;">Scan errors: ${d.scan_errors.join(" | ")}</div>`;
        }

        html += `<div style="margin-top:8px;font-size:0.76rem;color:var(--text-secondary);">Generated: ${d.generated_at_ist || "-"} | Source: ${d.source || "-"}</div>`;
        el.innerHTML = html;
    } catch (err) {
        el.innerHTML = `<span class="empty-msg">Error: ${err.message}</span>`;
    }
}

// ---- Portfolio Card ----
function toNumber(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

function formatAxisTime(ts) {
    if (!ts) return "";
    const t = String(ts);
    return t.length >= 16 ? t.slice(5, 16) : t;
}

function buildChartSvg(values, color, { fill = false, forceTopZero = false } = {}) {
    const n = values.length;
    const width = 360;
    const height = 150;
    const pad = 14;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2;

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const minY = forceTopZero ? Math.min(minValue, 0) : minValue;
    const maxY = forceTopZero ? 0 : maxValue;
    const range = Math.max(maxY - minY, 1e-9);

    const points = values.map((v, i) => {
        const x = pad + (i * plotW) / Math.max(n - 1, 1);
        const y = pad + ((maxY - v) / range) * plotH;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const line = points.join(" ");
    const baselineY = pad + ((maxY - minY) / range) * plotH;
    const area = `${points[0]} ${line} ${points[n - 1].split(",")[0]},${baselineY.toFixed(2)} ${points[0].split(",")[0]},${baselineY.toFixed(2)}`;

    return `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
            <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" class="chart-grid-line"></line>
            <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" class="chart-grid-line"></line>
            ${fill ? `<polygon points="${area}" fill="${color}" fill-opacity="0.15"></polygon>` : ""}
            <polyline points="${line}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline>
        </svg>
    `;
}

function renderPortfolioCharts(curve) {
    const host = document.getElementById("portfolioCharts");
    if (!host) return;

    if (!Array.isArray(curve) || curve.length < 2) {
        host.innerHTML = '<div class="chart-empty">Need at least 2 equity points to plot charts.</div>';
        return;
    }

    const equity = curve.map((pt) => toNumber(pt.portfolio_value));
    const drawdown = curve.map((pt) => Math.min(0, toNumber(pt.drawdown_pct)));
    const labels = curve.map((pt) => formatAxisTime(pt.timestamp));

    const last = curve[curve.length - 1];
    const minDd = Math.min(...drawdown);

    host.innerHTML = `
        <div class="chart-panel">
            <div class="chart-head">
                <span class="chart-title">Equity Curve</span>
                <span class="chart-stat">Latest: INR ${toNumber(last.portfolio_value).toLocaleString("en-IN")}</span>
            </div>
            <div class="chart-body">${buildChartSvg(equity, "var(--accent-blue)", { fill: true })}</div>
            <div class="chart-axis">
                <span>${labels[0] || "-"}</span>
                <span>${labels[Math.floor((labels.length - 1) / 2)] || "-"}</span>
                <span>${labels[labels.length - 1] || "-"}</span>
            </div>
        </div>
        <div class="chart-panel">
            <div class="chart-head">
                <span class="chart-title">Drawdown %</span>
                <span class="chart-stat">Worst: ${minDd.toFixed(2)}%</span>
            </div>
            <div class="chart-body">${buildChartSvg(drawdown, "var(--accent-red)", { fill: true, forceTopZero: true })}</div>
            <div class="chart-axis">
                <span>${labels[0] || "-"}</span>
                <span>${labels[Math.floor((labels.length - 1) / 2)] || "-"}</span>
                <span>${labels[labels.length - 1] || "-"}</span>
            </div>
        </div>
    `;
}

async function loadPortfolio() {
    const el = document.getElementById("portfolioContent");
    const chartEl = document.getElementById("portfolioCharts");
    el.innerHTML = '<span class="loading">Loading portfolio...</span>';
    if (chartEl) chartEl.innerHTML = "";
    try {
        const res = await fetch(`${API}/api/portfolio`);
        const d = await res.json();
        let html = `
            <div class="metric-row"><span class="label">Cash</span><span class="value">INR ${Number(d.cash).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Invested</span><span class="value">INR ${Number(d.total_invested).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Portfolio Value</span><span class="value">INR ${Number(d.portfolio_value).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Net Profit</span><span class="value" style="color:${d.net_profit_inr >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">INR ${Number(d.net_profit_inr).toLocaleString("en-IN")} (${Number(d.net_profit_pct).toFixed(2)}%)</span></div>
            <div class="metric-row"><span class="label">Max Drawdown</span><span class="value" style="color:var(--accent-red)">${Number(d.max_drawdown_pct).toFixed(2)}%</span></div>
            <div class="metric-row"><span class="label">Realized P&L</span><span class="value" style="color:${d.realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">INR ${Number(d.realized_pnl).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Unrealized P&L</span><span class="value" style="color:${d.unrealized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">INR ${Number(d.unrealized_pnl).toLocaleString("en-IN")}</span></div>
        `;
        if (d.open_positions && d.open_positions.length > 0) {
            html += `<h3 style="margin-top:14px;">Open Positions</h3><div class="positions-list">`;
            for (const p of d.open_positions) {
                html += `
                    <div class="position-item">
                        <span class="sym">${p.symbol}</span>
                        Qty: ${p.qty} | Entry: ${p.entry} | Stop: ${p.stop} | Target: ${p.target}
                    </div>`;
            }
            html += `</div>`;
        } else {
            html += `<div class="empty-msg" style="margin-top:10px;">No open positions.</div>`;
        }
        if (d.quote_errors && d.quote_errors.length > 0) {
            html += `<div class="empty-msg" style="margin-top:10px;">Quote errors: ${d.quote_errors.join(" | ")}</div>`;
        }
        el.innerHTML = html;
        renderPortfolioCharts(d.recent_equity_curve || []);
    } catch (err) {
        el.innerHTML = `<span class="empty-msg">Error: ${err.message}</span>`;
        if (chartEl) chartEl.innerHTML = "";
    }
}

// ---- Init ----
document.addEventListener("DOMContentLoaded", () => {
    loadRegime();
    loadSignalBoard();
    loadPortfolio();
    chatInput.focus();
});
