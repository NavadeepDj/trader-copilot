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

// ---- Portfolio Card ----
async function loadPortfolio() {
    const el = document.getElementById("portfolioContent");
    el.innerHTML = '<span class="loading">Loading portfolio...</span>';
    try {
        const res = await fetch(`${API}/api/portfolio`);
        const d = await res.json();
        let html = `
            <div class="metric-row"><span class="label">Cash</span><span class="value">INR ${Number(d.cash).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Invested</span><span class="value">INR ${Number(d.total_invested).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Portfolio Value</span><span class="value">INR ${Number(d.portfolio_value).toLocaleString("en-IN")}</span></div>
            <div class="metric-row"><span class="label">Realized P&L</span><span class="value" style="color:${d.realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">INR ${Number(d.realized_pnl).toLocaleString("en-IN")}</span></div>
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
        el.innerHTML = html;
    } catch (err) {
        el.innerHTML = `<span class="empty-msg">Error: ${err.message}</span>`;
    }
}

// ---- Init ----
document.addEventListener("DOMContentLoaded", () => {
    loadRegime();
    loadPortfolio();
    chatInput.focus();
});
