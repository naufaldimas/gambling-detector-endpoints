/**
 * Gambling Detector — Monitoring Dashboard (Frontend)
 * =====================================================
 *
 * Pure vanilla JS — no dependencies.
 * Handles: API calls, page routing, tables, search, filters, sorting,
 *          activity feed, stats rendering, and modal detail views.
 */

// ─── State ──────────────────────────────────────────────────────────────────

const state = {
  currentPage: 'dashboard',
  stats: null,
  devices: [],
  websites: [],
  activities: [],
  datasetResults: [],
  datasetTotalCount: 0,
  datasetCurrentPage: 1,
  datasetPerPage: 100,

  // Filters
  dashDeviceFilter: 'all',
  activityFilter: 'all',
  devicesPageFilter: 'all',
  websitesPageStatusFilter: 'all',
  websitesPageCatFilter: 'all',
  datasetLabelFilter: '',

  // Sort
  devicesSort: { field: 'last_seen', order: 'desc' },
  websitesSort: { field: 'detected_at', order: 'desc' },

  // Search
  devicesSearchQuery: '',
  websitesSearchQuery: '',
  datasetSearchQuery: '',
  globalSearchQuery: '',

  // Loading states
  isLoading: true,
  hasError: false,
};

// ─── API Helpers ────────────────────────────────────────────────────────────

const API_BASE = '/api';

async function apiFetch(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

// ─── Data Fetching ──────────────────────────────────────────────────────────

async function fetchStats() {
  const data = await apiFetch('/stats');
  state.stats = data;
  return data;
}

async function fetchDevices(params = {}) {
  const query = new URLSearchParams(params).toString();
  const data = await apiFetch(`/devices?${query}`);
  state.devices = data.devices;
  return data;
}

async function fetchWebsites(params = {}) {
  const query = new URLSearchParams(params).toString();
  const data = await apiFetch(`/websites?${query}`);
  state.websites = data.websites;
  return data;
}

async function fetchActivities(params = {}) {
  const query = new URLSearchParams(params).toString();
  const data = await apiFetch(`/activities?${query}`);
  state.activities = data.activities;
  return data;
}

async function searchDataset(params = {}) {
  const query = new URLSearchParams(params).toString();
  const data = await apiFetch(`/dataset/search?${query}`);
  state.datasetResults = data.results;
  state.datasetTotalCount = data.total;
  return data;
}

async function updateWebsiteStatus(id, newStatus) {
  return await apiFetch(`/websites/${id}/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: newStatus }),
  });
}

// ─── Rendering: Stats Cards ────────────────────────────────────────────────

function renderStats() {
  const s = state.stats;
  if (!s) return;

  const grid = document.getElementById('statsGrid');
  grid.innerHTML = `
    <div class="stat-card accent-indigo">
      <div class="stat-card-header">
        <span class="stat-card-label">Total Devices</span>
        <div class="stat-card-icon">📱</div>
      </div>
      <div class="stat-card-value">${s.devices.total}</div>
      <div class="stat-card-sub"><span class="highlight">${s.devices.online} online</span> · ${s.devices.idle} idle · ${s.devices.offline} offline</div>
    </div>

    <div class="stat-card accent-green">
      <div class="stat-card-header">
        <span class="stat-card-label">Device Aktif</span>
        <div class="stat-card-icon">✅</div>
      </div>
      <div class="stat-card-value">${s.devices.online}</div>
      <div class="stat-card-sub">VPN aktif & terhubung</div>
    </div>

    <div class="stat-card accent-amber">
      <div class="stat-card-header">
        <span class="stat-card-label">Website Baru</span>
        <div class="stat-card-icon">🌐</div>
      </div>
      <div class="stat-card-value">${s.websites.total_new}</div>
      <div class="stat-card-sub"><span class="warning">${s.websites.pending_review} perlu review</span></div>
    </div>

    <div class="stat-card accent-rose">
      <div class="stat-card-header">
        <span class="stat-card-label">Total Blocked</span>
        <div class="stat-card-icon">🚫</div>
      </div>
      <div class="stat-card-value">${formatNumber(s.queries.blocked)}</div>
      <div class="stat-card-sub">${s.queries.block_rate}% dari ${formatNumber(s.queries.total)} query</div>
    </div>

    <div class="stat-card accent-cyan">
      <div class="stat-card-header">
        <span class="stat-card-label">Dataset Size</span>
        <div class="stat-card-icon">📚</div>
      </div>
      <div class="stat-card-value">${formatNumber(s.dataset.total)}</div>
      <div class="stat-card-sub">${formatNumber(s.dataset.gambling)} gambling · ${formatNumber(s.dataset.legitimate)} legit</div>
    </div>

    <div class="stat-card accent-violet">
      <div class="stat-card-header">
        <span class="stat-card-label">Model Threshold</span>
        <div class="stat-card-icon">🧠</div>
      </div>
      <div class="stat-card-value">${s.model.threshold ?? 'N/A'}</div>
      <div class="stat-card-sub">F1: ${s.model.f1_score ?? 'N/A'} · ${s.model.n_features ?? '?'} features</div>
    </div>
  `;
}

// ─── Rendering: Device Tables ──────────────────────────────────────────────

function renderDashDevices() {
  let devices = state.devices;
  if (state.dashDeviceFilter !== 'all') {
    devices = devices.filter(d => d.status === state.dashDeviceFilter);
  }

  document.getElementById('dashDeviceCount').textContent = devices.length;
  const tbody = document.getElementById('dashDevicesBody');

  if (devices.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7"><div class="state-container"><div class="state-icon">📱</div><div class="state-title">Tidak ada device</div><div class="state-text">Tidak ada device yang sesuai filter</div></div></td></tr>`;
    return;
  }

  tbody.innerHTML = devices.map(d => `
    <tr>
      <td class="cell-primary">${escapeHtml(d.name)}</td>
      <td class="cell-mono">${escapeHtml(d.ip)}</td>
      <td>${statusBadge(d.status)}</td>
      <td>${vpnIndicator(d.vpn_active)}</td>
      <td>${formatNumber(d.total_queries)}</td>
      <td><span style="color: var(--cat-gambling)">${d.blocked_queries}</span></td>
      <td class="cell-mono">${timeAgo(d.last_seen)}</td>
    </tr>
  `).join('');
}

function renderDevicesPage() {
  let devices = state.devices;
  if (state.devicesPageFilter !== 'all') {
    devices = devices.filter(d => d.status === state.devicesPageFilter);
  }
  if (state.devicesSearchQuery) {
    const q = state.devicesSearchQuery.toLowerCase();
    devices = devices.filter(d =>
      d.name.toLowerCase().includes(q) ||
      d.ip.toLowerCase().includes(q) ||
      d.mac.toLowerCase().includes(q) ||
      (d.last_domain || '').toLowerCase().includes(q)
    );
  }

  // Sort
  const { field, order } = state.devicesSort;
  devices = [...devices].sort((a, b) => {
    let va = a[field] ?? '', vb = b[field] ?? '';
    if (typeof va === 'number' && typeof vb === 'number') {
      return order === 'asc' ? va - vb : vb - va;
    }
    va = String(va).toLowerCase();
    vb = String(vb).toLowerCase();
    return order === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  });

  document.getElementById('devicesPageCount').textContent = devices.length;
  const tbody = document.getElementById('devicesPageBody');

  if (devices.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11"><div class="state-container"><div class="state-icon">📱</div><div class="state-title">Tidak ada device ditemukan</div><div class="state-text">Coba ubah filter atau kata kunci pencarian</div></div></td></tr>`;
    return;
  }

  tbody.innerHTML = devices.map(d => `
    <tr style="cursor:pointer" onclick="showDeviceDetail('${escapeHtml(d.id)}')">
      <td class="cell-primary">${escapeHtml(d.name)}</td>
      <td class="cell-mono">${escapeHtml(d.ip)}</td>
      <td class="cell-mono">${escapeHtml(d.mac)}</td>
      <td>${statusBadge(d.status)}</td>
      <td>${vpnIndicator(d.vpn_active)}</td>
      <td>${escapeHtml(d.os)}</td>
      <td>${escapeHtml(d.app_version)}</td>
      <td>${formatNumber(d.total_queries)}</td>
      <td><span style="color: var(--cat-gambling)">${d.blocked_queries}</span></td>
      <td class="cell-domain">${escapeHtml(d.last_domain || '-')}</td>
      <td class="cell-mono">${timeAgo(d.last_seen)}</td>
    </tr>
  `).join('');

  // Update sort indicators
  document.querySelectorAll('#devicesPageTable th').forEach(th => {
    th.classList.toggle('sorted', th.dataset.sort === field);
    const icon = th.querySelector('.sort-icon');
    if (icon && th.dataset.sort === field) {
      icon.textContent = order === 'asc' ? '↑' : '↓';
    } else if (icon) {
      icon.textContent = '↕';
    }
  });
}

// ─── Rendering: Websites Table ─────────────────────────────────────────────

function renderWebsitesPage() {
  let websites = state.websites;
  if (state.websitesPageStatusFilter !== 'all') {
    websites = websites.filter(w => w.status === state.websitesPageStatusFilter);
  }
  if (state.websitesPageCatFilter !== 'all') {
    websites = websites.filter(w => w.category === state.websitesPageCatFilter);
  }
  if (state.websitesSearchQuery) {
    const q = state.websitesSearchQuery.toLowerCase();
    websites = websites.filter(w =>
      w.domain.toLowerCase().includes(q) ||
      (w.title || '').toLowerCase().includes(q) ||
      (w.source_device || '').toLowerCase().includes(q)
    );
  }

  // Sort
  const { field, order } = state.websitesSort;
  websites = [...websites].sort((a, b) => {
    let va = a[field] ?? '', vb = b[field] ?? '';
    if (typeof va === 'number' && typeof vb === 'number') {
      return order === 'asc' ? va - vb : vb - va;
    }
    va = String(va).toLowerCase();
    vb = String(vb).toLowerCase();
    return order === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  });

  document.getElementById('websitesPageCount').textContent = websites.length;
  const tbody = document.getElementById('websitesPageBody');

  if (websites.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="state-container"><div class="state-icon">🌐</div><div class="state-title">Tidak ada website ditemukan</div><div class="state-text">Coba ubah filter atau kata kunci pencarian</div></div></td></tr>`;
    return;
  }

  tbody.innerHTML = websites.map(w => `
    <tr>
      <td class="cell-domain">${escapeHtml(w.domain)}</td>
      <td>${escapeHtml(truncate(w.title || '-', 40))}</td>
      <td class="cell-primary">${escapeHtml(w.source_device)}</td>
      <td>${scoreBar(w.ml_score)}</td>
      <td>${categoryBadge(w.category)}</td>
      <td>${websiteStatusBadge(w.status)}</td>
      <td class="cell-mono">${timeAgo(w.detected_at)}</td>
      <td>
        <div class="action-btn-group">
          <button class="action-btn" onclick="showWebsiteDetail('${escapeHtml(w.id)}')">👁 Detail</button>
          ${w.status === 'pending_review' ? `
            <button class="action-btn btn-success" onclick="markWebsite('${escapeHtml(w.id)}', 'added_to_dataset')">📥 Add</button>
            <button class="action-btn btn-danger" onclick="markWebsite('${escapeHtml(w.id)}', 'dismissed')">✕</button>
          ` : ''}
        </div>
      </td>
    </tr>
  `).join('');

  // Update sort indicators
  document.querySelectorAll('#websitesPageTable th').forEach(th => {
    th.classList.toggle('sorted', th.dataset.sort === field);
    const icon = th.querySelector('.sort-icon');
    if (icon && th.dataset.sort === field) {
      icon.textContent = order === 'asc' ? '↑' : '↓';
    } else if (icon) {
      icon.textContent = '↕';
    }
  });
}

// ─── Rendering: Activity Feed ──────────────────────────────────────────────

function renderActivities() {
  let activities = state.activities;
  if (state.activityFilter !== 'all') {
    activities = activities.filter(a => a.type === state.activityFilter);
  }

  const feed = document.getElementById('activityFeed');

  if (activities.length === 0) {
    feed.innerHTML = `<div class="state-container"><div class="state-icon">⚡</div><div class="state-title">Belum ada aktivitas</div></div>`;
    return;
  }

  feed.innerHTML = activities.map(a => {
    const icon = activityIcon(a.type);
    const text = activityText(a);
    return `
      <div class="activity-item">
        <div class="activity-icon ${a.type}">${icon}</div>
        <div class="activity-content">
          <div class="activity-text">${text}</div>
          <div class="activity-time">${timeAgo(a.time)}</div>
        </div>
      </div>
    `;
  }).join('');
}

// ─── Rendering: Info Panels ────────────────────────────────────────────────

function renderInfoPanels() {
  const s = state.stats;
  if (!s) return;

  // Alerts
  const alertsList = document.getElementById('alertsList');
  const alerts = [];

  if (s.devices.offline > 0) {
    alerts.push({ label: `${s.devices.offline} device offline`, value: '⚠️', cls: 'warning' });
  }
  if (s.websites.pending_review > 0) {
    alerts.push({ label: `${s.websites.pending_review} website perlu review`, value: '⏳', cls: 'warning' });
  }

  // Check for devices not seen in a long time
  const longInactive = state.devices.filter(d => {
    const diff = Date.now() - new Date(d.last_seen).getTime();
    return diff > 48 * 60 * 60 * 1000; // 48 hours
  });
  if (longInactive.length > 0) {
    alerts.push({ label: `${longInactive.length} device tidak aktif >48 jam`, value: '🔴', cls: '' });
  }

  if (alerts.length === 0) {
    alerts.push({ label: 'Semua normal', value: '✅', cls: '' });
  }

  alertsList.innerHTML = alerts.map(a => `
    <li><span class="label">${a.label}</span><span class="value">${a.value}</span></li>
  `).join('');

  // Top blocked domains
  const topBlockedList = document.getElementById('topBlockedList');
  const blockActivities = state.activities.filter(a => a.type === 'block');
  const domainCounts = {};
  blockActivities.forEach(a => {
    domainCounts[a.domain] = (domainCounts[a.domain] || 0) + 1;
  });
  const topDomains = Object.entries(domainCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  topBlockedList.innerHTML = topDomains.length > 0
    ? topDomains.map(([domain, count]) => `
        <li><span class="label" style="color: var(--accent-cyan)">${escapeHtml(domain)}</span><span class="value">${count}×</span></li>
      `).join('')
    : '<li><span class="label">Belum ada data</span></li>';

  // Model info
  const modelInfoList = document.getElementById('modelInfoList');
  modelInfoList.innerHTML = `
    <li><span class="label">Threshold</span><span class="value">${s.model.threshold ?? '-'}</span></li>
    <li><span class="label">F1-Score (domain-only)</span><span class="value">${s.model.f1_score ?? '-'}</span></li>
    <li><span class="label">Features</span><span class="value">${s.model.n_features ?? '-'}</span></li>
    <li><span class="label">Dataset Size</span><span class="value">${formatNumber(s.dataset.total)}</span></li>
    <li><span class="label">Gambling Samples</span><span class="value">${formatNumber(s.dataset.gambling)}</span></li>
    <li><span class="label">Legitimate Samples</span><span class="value">${formatNumber(s.dataset.legitimate)}</span></li>
  `;
}

// ─── Rendering: Dataset Search ─────────────────────────────────────────────

function renderDatasetResults() {
  const tbody = document.getElementById('datasetBody');
  const results = state.datasetResults;
  const total = state.datasetTotalCount;
  const showing = results.length;

  // Update badge with showing/total info
  const countEl = document.getElementById('datasetResultCount');
  if (total > 0) {
    countEl.textContent = showing < total ? `${showing} / ${formatNumber(total)}` : formatNumber(total);
  } else {
    countEl.textContent = '0';
  }

  if (results.length === 0 && !state.datasetSearchQuery && !state.datasetLabelFilter) {
    tbody.innerHTML = `<tr><td colspan="4"><div class="state-container"><div class="state-icon">📭</div><div class="state-title">Dataset kosong</div><div class="state-text">Tidak ada data ditemukan di dataset</div></div></td></tr>`;
    return;
  }

  if (results.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4"><div class="state-container"><div class="state-icon">📭</div><div class="state-title">Tidak ditemukan</div><div class="state-text">Coba kata kunci lain atau ubah filter label</div></div></td></tr>`;
    return;
  }

  let html = results.map(r => `
    <tr>
      <td class="cell-mono">${escapeHtml(r.id)}</td>
      <td class="cell-domain" style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-tooltip="${escapeHtml(r.url)}">${escapeHtml(r.url)}</td>
      <td>${r.label === 1 ? '<span class="badge badge-gambling">🎰 Gambling</span>' : '<span class="badge badge-legitimate">✅ Legit</span>'}</td>
      <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(truncate(r.title, 60))}</td>
    </tr>
  `).join('');

  // Pagination controls
  if (total > 0) {
    const totalPages = Math.ceil(total / state.datasetPerPage);
    const currentPage = state.datasetCurrentPage;
    const hasPrev = currentPage > 1;
    const hasNext = currentPage < totalPages;

    html += `
      <tr>
        <td colspan="4">
          <div class="pagination-container" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 15px;">
            <div class="pagination-info" style="color: var(--text-tertiary); font-size: 0.85rem;">
              Menampilkan ${(currentPage - 1) * state.datasetPerPage + 1} - ${Math.min(currentPage * state.datasetPerPage, total)} dari ${formatNumber(total)} data
            </div>
            <div class="pagination-controls" style="display: flex; gap: 10px; align-items: center;">
              <button class="action-btn" onclick="changeDatasetPage(${currentPage - 1})" ${!hasPrev ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>&laquo; Prev</button>
              <span style="font-size: 0.9rem; font-weight: 500;">Halaman ${currentPage} / ${totalPages}</span>
              <button class="action-btn" onclick="changeDatasetPage(${currentPage + 1})" ${!hasNext ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>Next &raquo;</button>
            </div>
          </div>
        </td>
      </tr>
    `;
  }

  tbody.innerHTML = html;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatNumber(n) {
  if (n == null) return '-';
  return n.toLocaleString('id-ID');
}

function truncate(str, maxLen) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
}

function timeAgo(isoString) {
  if (!isoString) return '-';
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return 'Baru saja';
  if (diffMin < 60) return `${diffMin} menit lalu`;
  if (diffHour < 24) return `${diffHour} jam lalu`;
  if (diffDay < 7) return `${diffDay} hari lalu`;
  return date.toLocaleDateString('id-ID', { day: 'numeric', month: 'short', year: 'numeric' });
}

function statusBadge(status) {
  const map = {
    online: '<span class="badge badge-online">Online</span>',
    idle: '<span class="badge badge-idle">Idle</span>',
    offline: '<span class="badge badge-offline">Offline</span>',
  };
  return map[status] || `<span class="badge">${status}</span>`;
}

function vpnIndicator(active) {
  return active
    ? '<span class="vpn-indicator active">🔒 Active</span>'
    : '<span class="vpn-indicator inactive">🔓 Off</span>';
}

function categoryBadge(cat) {
  const map = {
    gambling: '<span class="badge badge-gambling">🎰 Gambling</span>',
    legitimate: '<span class="badge badge-legitimate">✅ Legit</span>',
  };
  return map[cat] || `<span class="badge">${cat}</span>`;
}

function websiteStatusBadge(status) {
  const map = {
    pending_review: '<span class="badge badge-pending">⏳ Pending</span>',
    added_to_dataset: '<span class="badge badge-added">📥 Added</span>',
    verified: '<span class="badge badge-verified">✅ Verified</span>',
    dismissed: '<span class="badge badge-dismissed">✕ Dismissed</span>',
  };
  return map[status] || `<span class="badge">${status}</span>`;
}

function scoreBar(score) {
  const pct = Math.round(score * 100);
  let cls = 'score-low';
  if (score >= 0.7) cls = 'score-high';
  else if (score >= 0.4) cls = 'score-medium';

  return `
    <div class="score-bar ${cls}">
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="score-value">${score.toFixed(2)}</span>
    </div>
  `;
}

function activityIcon(type) {
  const map = { block: '🚫', allow: '✅', device_connect: '📱', device_disconnect: '📴' };
  return map[type] || '●';
}

function activityText(a) {
  switch (a.type) {
    case 'block':
      return `<strong>${escapeHtml(a.device)}</strong> diblokir mengakses <span class="domain-name">${escapeHtml(a.domain)}</span> <span style="color:var(--cat-gambling)">(${a.score.toFixed(2)})</span>`;
    case 'allow':
      return `<strong>${escapeHtml(a.device)}</strong> mengakses <span class="domain-name">${escapeHtml(a.domain)}</span>`;
    case 'device_connect':
      return `<strong>${escapeHtml(a.device)}</strong> terhubung ke jaringan`;
    case 'device_disconnect':
      return `<strong>${escapeHtml(a.device)}</strong> terputus dari jaringan`;
    default:
      return escapeHtml(JSON.stringify(a));
  }
}

// ─── Modal ──────────────────────────────────────────────────────────────────

function openModal(title, bodyHtml) {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalBody').innerHTML = bodyHtml;
  document.getElementById('modalOverlay').classList.add('active');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('active');
}

function showDeviceDetail(deviceId) {
  const d = state.devices.find(dev => dev.id === deviceId);
  if (!d) return;

  openModal(`📱 ${d.name}`, `
    <div class="detail-row"><span class="detail-label">Device ID</span><span class="detail-value">${escapeHtml(d.id)}</span></div>
    <div class="detail-row"><span class="detail-label">Model</span><span class="detail-value">${escapeHtml(d.model)}</span></div>
    <div class="detail-row"><span class="detail-label">IP Address</span><span class="detail-value">${escapeHtml(d.ip)}</span></div>
    <div class="detail-row"><span class="detail-label">MAC Address</span><span class="detail-value">${escapeHtml(d.mac)}</span></div>
    <div class="detail-row"><span class="detail-label">OS</span><span class="detail-value">${escapeHtml(d.os)}</span></div>
    <div class="detail-row"><span class="detail-label">App Version</span><span class="detail-value">${escapeHtml(d.app_version)}</span></div>
    <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value">${statusBadge(d.status)}</span></div>
    <div class="detail-row"><span class="detail-label">VPN</span><span class="detail-value">${vpnIndicator(d.vpn_active)}</span></div>
    <div class="detail-row"><span class="detail-label">Total Queries</span><span class="detail-value">${formatNumber(d.total_queries)}</span></div>
    <div class="detail-row"><span class="detail-label">Blocked Queries</span><span class="detail-value" style="color:var(--cat-gambling)">${d.blocked_queries}</span></div>
    <div class="detail-row"><span class="detail-label">Domain Terakhir</span><span class="detail-value" style="color:var(--accent-cyan)">${escapeHtml(d.last_domain || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">Terakhir Aktif</span><span class="detail-value">${timeAgo(d.last_seen)}</span></div>
  `);
}

function showWebsiteDetail(websiteId) {
  const w = state.websites.find(web => web.id === websiteId);
  if (!w) return;

  openModal(`🌐 ${w.domain}`, `
    <div class="detail-row"><span class="detail-label">Domain</span><span class="detail-value" style="color:var(--accent-cyan)">${escapeHtml(w.domain)}</span></div>
    <div class="detail-row"><span class="detail-label">Full URL</span><span class="detail-value">${escapeHtml(w.full_url)}</span></div>
    <div class="detail-row"><span class="detail-label">Judul</span><span class="detail-value">${escapeHtml(w.title || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">ML Score</span><span class="detail-value">${scoreBar(w.ml_score)}</span></div>
    <div class="detail-row"><span class="detail-label">Kategori</span><span class="detail-value">${categoryBadge(w.category)}</span></div>
    <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value">${websiteStatusBadge(w.status)}</span></div>
    <div class="detail-row"><span class="detail-label">Device Asal</span><span class="detail-value">${escapeHtml(w.source_device)}</span></div>
    <div class="detail-row"><span class="detail-label">Terdeteksi</span><span class="detail-value">${timeAgo(w.detected_at)}</span></div>
  `);
}

// ─── Website Actions ────────────────────────────────────────────────────────

async function markWebsite(id, newStatus) {
  try {
    await updateWebsiteStatus(id, newStatus);
    // Re-fetch websites to update UI
    await fetchWebsites();
    renderWebsitesPage();
    // Also refresh stats
    await fetchStats();
    renderStats();
    renderInfoPanels();
  } catch (err) {
    console.error('Failed to update website status:', err);
  }
}

// ─── Page Navigation ────────────────────────────────────────────────────────

function navigateTo(page) {
  state.currentPage = page;

  // Update nav links
  document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.page === page);
  });

  // Show/hide pages
  document.querySelectorAll('.page-section').forEach(section => {
    section.classList.toggle('active', section.id === `page-${page}`);
  });

  // Close mobile nav
  document.getElementById('navbarNav').classList.remove('open');

  // Refresh data for the page
  refreshCurrentPage();
  refreshStatsHeader(); // Update the top stats too
}

async function refreshStatsHeader() {
  await fetchStats();
  renderStats();
  renderInfoPanels();
}

async function refreshCurrentPage() {
  switch (state.currentPage) {
    case 'dashboard':
      await Promise.all([fetchDevices(), fetchActivities()]);
      renderDashDevices();
      renderActivities();
      break;
    case 'devices':
      await fetchDevices();
      renderDevicesPage();
      break;
    case 'websites':
      await fetchWebsites();
      renderWebsitesPage();
      break;
    case 'dataset':
      doDatasetSearch();
      break;
  }
}

// ─── Event Listeners ────────────────────────────────────────────────────────

function initEventListeners() {
  // Nav links
  document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
    link.addEventListener('click', () => navigateTo(link.dataset.page));
  });

  // Navbar toggle (mobile)
  document.getElementById('navbarToggle').addEventListener('click', () => {
    document.getElementById('navbarNav').classList.toggle('open');
  });

  // Modal close
  document.getElementById('modalClose').addEventListener('click', closeModal);
  document.getElementById('modalOverlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  // Filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      const filter = btn.dataset.filter;

      // Update active state within same group
      btn.closest('.filter-group').querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Apply filter based on target
      switch (target) {
        case 'dash-devices':
          state.dashDeviceFilter = filter;
          renderDashDevices();
          break;
        case 'activities':
          state.activityFilter = filter;
          renderActivities();
          break;
        case 'devices-page':
          state.devicesPageFilter = filter;
          renderDevicesPage();
          break;
        case 'websites-page-status':
          state.websitesPageStatusFilter = filter;
          renderWebsitesPage();
          break;
        case 'websites-page-cat':
          state.websitesPageCatFilter = filter;
          renderWebsitesPage();
          break;
        case 'dataset-label':
          state.datasetLabelFilter = filter;
          doDatasetSearch();
          break;
      }
    });
  });

  // Sorting (devices page)
  document.querySelectorAll('#devicesPageTable th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (state.devicesSort.field === field) {
        state.devicesSort.order = state.devicesSort.order === 'asc' ? 'desc' : 'asc';
      } else {
        state.devicesSort = { field, order: 'asc' };
      }
      renderDevicesPage();
    });
  });

  // Sorting (websites page)
  document.querySelectorAll('#websitesPageTable th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (state.websitesSort.field === field) {
        state.websitesSort.order = state.websitesSort.order === 'asc' ? 'desc' : 'asc';
      } else {
        state.websitesSort = { field, order: 'asc' };
      }
      renderWebsitesPage();
    });
  });

  // Search: Devices page
  document.getElementById('devicesSearch').addEventListener('input', debounce((e) => {
    state.devicesSearchQuery = e.target.value;
    renderDevicesPage();
  }, 250));

  // Search: Websites page
  document.getElementById('websitesSearch').addEventListener('input', debounce((e) => {
    state.websitesSearchQuery = e.target.value;
    renderWebsitesPage();
  }, 250));

  // Search: Dataset page
  document.getElementById('datasetSearchInput').addEventListener('input', debounce((e) => {
    state.datasetSearchQuery = e.target.value;
    doDatasetSearch();
  }, 400));

  // Global search
  document.getElementById('globalSearch').addEventListener('input', debounce((e) => {
    state.globalSearchQuery = e.target.value;
    handleGlobalSearch(e.target.value);
  }, 300));

  document.getElementById('globalSearch').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      handleGlobalSearch(e.target.value);
    }
  });
}

function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

// ─── Search Handlers ────────────────────────────────────────────────────────

async function doDatasetSearch() {
  const q = state.datasetSearchQuery;
  const label = state.datasetLabelFilter;

  // Reset pagination on new search
  state.datasetCurrentPage = 1;

  const tbody = document.getElementById('datasetBody');
  tbody.innerHTML = `<tr><td colspan="4"><div class="state-container"><div class="loading-spinner"></div><div class="state-title">Memuat data...</div></div></td></tr>`;

  try {
    const limit = state.datasetPerPage;
    const offset = 0;
    await searchDataset({ q, label, limit, offset });
    renderDatasetResults();
  } catch {
    tbody.innerHTML = `<tr><td colspan="4"><div class="state-container error-state"><div class="state-icon">❌</div><div class="state-title">Gagal memuat data</div><div class="state-text">Periksa koneksi ke server</div><button class="retry-btn" onclick="doDatasetSearch()">Coba Lagi</button></div></td></tr>`;
  }
}

async function changeDatasetPage(newPage) {
  const q = state.datasetSearchQuery;
  const label = state.datasetLabelFilter;
  const totalPages = Math.ceil(state.datasetTotalCount / state.datasetPerPage);

  if (newPage < 1 || newPage > totalPages) return;

  state.datasetCurrentPage = newPage;
  const limit = state.datasetPerPage;
  const offset = (newPage - 1) * limit;

  const tbody = document.getElementById('datasetBody');
  tbody.innerHTML = `<tr><td colspan="4"><div class="state-container"><div class="loading-spinner"></div><div class="state-title">Memuat Halaman ${newPage}...</div></div></td></tr>`;

  try {
    await searchDataset({ q, label, limit, offset });
    renderDatasetResults();
  } catch {
    tbody.innerHTML = `<tr><td colspan="4"><div class="state-container error-state"><div class="state-icon">❌</div><div class="state-title">Gagal memuat data</div><div class="state-text">Periksa koneksi ke server</div><button class="retry-btn" onclick="changeDatasetPage(${newPage})">Coba Lagi</button></div></td></tr>`;
  }
}

function handleGlobalSearch(query) {
  if (!query.trim()) return;

  // Smart routing: detect what the search might be for
  const q = query.trim().toLowerCase();

  // Check if it looks like an IP address or MAC
  if (/^\d{1,3}\./.test(q) || /^[0-9a-f]{2}:/.test(q)) {
    state.devicesSearchQuery = q;
    document.getElementById('devicesSearch').value = q;
    navigateTo('devices');
    return;
  }

  // Check if it looks like a domain
  if (q.includes('.') && !q.includes(' ')) {
    state.websitesSearchQuery = q;
    document.getElementById('websitesSearch').value = q;
    navigateTo('websites');
    return;
  }

  // Default: search dataset
  state.datasetSearchQuery = q;
  document.getElementById('datasetSearchInput').value = q;
  navigateTo('dataset');
  doDatasetSearch();
}

// ─── Initialization ─────────────────────────────────────────────────────────

async function init() {
  initEventListeners();

  try {
    // Parallel fetch all initial data
    await Promise.all([
      fetchStats(),
      fetchDevices(),
      fetchWebsites(),
      fetchActivities(),
    ]);

    state.isLoading = false;
    state.hasError = false;

    // Render everything
    renderStats();
    renderDashDevices();
    renderActivities();
    renderInfoPanels();

    // Update timestamp
    updateTimestamp();
  } catch (err) {
    state.hasError = true;
    console.error('Initialization failed:', err);
    document.getElementById('statsGrid').innerHTML = `
      <div class="stat-card" style="grid-column: 1 / -1">
        <div class="state-container error-state">
          <div class="state-icon">❌</div>
          <div class="state-title">Gagal memuat dashboard</div>
          <div class="state-text">Pastikan server backend berjalan di http://localhost:5000</div>
          <button class="retry-btn" onclick="init()">Coba Lagi</button>
        </div>
      </div>
    `;
  }
}

function updateTimestamp() {
  const el = document.getElementById('lastUpdated');
  const now = new Date();
  el.textContent = `Updated ${now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })}`;
}

// Auto-refresh every 30 seconds
setInterval(async () => {
  try {
    await fetchStats();
    renderStats();
    updateTimestamp();

    if (state.currentPage === 'dashboard') {
      await Promise.all([fetchDevices(), fetchActivities()]);
      renderDashDevices();
      renderActivities();
      renderInfoPanels();
    }
  } catch {
    // Silent fail on auto-refresh
  }
}, 30000);

// Start
document.addEventListener('DOMContentLoaded', init);
