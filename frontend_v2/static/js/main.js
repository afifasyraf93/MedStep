// MedStep v1.0

// ── Sidebar Toggle ────────────────────────────────────
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const wrapper = document.getElementById('mainWrapper');
    if (sidebar) sidebar.classList.toggle('collapsed');
    if (wrapper) wrapper.classList.toggle('expanded');
}

// ── Lightbox ──────────────────────────────────────────
function initLightbox() {
    const lightbox = document.createElement('div');
    lightbox.id = 'lightbox';
    lightbox.style.cssText = `
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.92);
        z-index: 99999;
        align-items: center;
        justify-content: center;
        cursor: zoom-out;
    `;
    lightbox.innerHTML = `
        <img id="lightboxImg" style="
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 8px;
            object-fit: contain;
            box-shadow: 0 8px 40px rgba(0,0,0,0.8);
        ">
        <button onclick="closeLightbox()" style="
            position: absolute;
            top: 20px; right: 24px;
            background: rgba(255,255,255,0.15);
            border: none;
            color: white;
            font-size: 1.5rem;
            width: 40px; height: 40px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        ">×</button>
    `;
    document.body.appendChild(lightbox);
    lightbox.addEventListener('click', e => {
        if (e.target === lightbox) closeLightbox();
    });
}

function openLightbox(src) {
    const lb = document.getElementById('lightbox');
    document.getElementById('lightboxImg').src = src;
    lb.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
    document.body.style.overflow = '';
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
});

// ── Severity ──────────────────────────────────────────
function getSeverity(prob) {
    if (prob >= 0.8) return { label: 'High', color: '#dc2626', bg: '#fee2e2' };
    if (prob >= 0.5) return { label: 'Moderate', color: '#d97706', bg: '#fef3c7' };
    return { label: 'Low', color: '#16a34a', bg: '#dcfce7' };
}

// ── Toast Notifications ───────────────────────────────
function initToasts() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
    `;
    document.body.appendChild(container);
}

function showToast(message, type = 'info', duration = 3500) {
    const colors = {
        success: { bg: '#dcfce7', border: '#4CAF7D', color: '#16a34a', icon: 'fa-circle-check' },
        error:   { bg: '#fee2e2', border: '#dc2626', color: '#dc2626', icon: 'fa-circle-exclamation' },
        info:    { bg: '#dbeafe', border: '#2E6B8A', color: '#2E6B8A', icon: 'fa-circle-info' },
        warning: { bg: '#fef3c7', border: '#d97706', color: '#d97706', icon: 'fa-triangle-exclamation' }
    };
    const c = colors[type] || colors.info;

    const toast = document.createElement('div');
    toast.style.cssText = `
        background: ${c.bg};
        border: 1px solid ${c.border};
        border-left: 4px solid ${c.border};
        color: ${c.color};
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 280px;
        max-width: 380px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        pointer-events: all;
        opacity: 0;
        transform: translateX(20px);
        transition: all 0.3s ease;
        cursor: pointer;
    `;
    toast.innerHTML = `
        <i class="fas ${c.icon}"></i>
        <span style="flex:1;">${message}</span>
        <i class="fas fa-times" style="opacity:0.5; font-size:0.75rem;"></i>
    `;
    toast.onclick = () => removeToast(toast);
    document.getElementById('toastContainer').appendChild(toast);

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        });
    });

    setTimeout(() => removeToast(toast), duration);
}

function removeToast(toast) {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
}

// ── API Connection Check ──────────────────────────────
async function checkAPIConnection() {
    const banner = document.createElement('div');
    banner.id = 'apiBanner';
    banner.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0;
        background: #fee2e2;
        border-bottom: 1px solid #fca5a5;
        color: #dc2626;
        padding: 10px 24px;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: center;
        z-index: 99998;
        display: none;
        align-items: center;
        justify-content: center;
        gap: 10px;
    `;
    banner.innerHTML = `
        <i class="fas fa-triangle-exclamation"></i>
        <span>API server is not responding. Please make sure the backend is running on port 8000.</span>
        <button onclick="retryConnection()" style="
            margin-left: 12px;
            padding: 4px 12px;
            border-radius: 6px;
            border: 1px solid #dc2626;
            background: transparent;
            color: #dc2626;
            font-size: 0.8rem;
            cursor: pointer;
        ">Retry</button>
    `;
    document.body.appendChild(banner);

    try {
        const res = await fetch('http://localhost:8000/health', {
            signal: AbortSignal.timeout(5000)
        });
        if (res.ok) {
            banner.style.display = 'none';
        } else {
            banner.style.display = 'flex';
        }
    } catch {
        banner.style.display = 'flex';
    }
}

async function retryConnection() {
    const banner = document.getElementById('apiBanner');
    banner.innerHTML = `
        <i class="fas fa-spinner fa-spin"></i>
        <span>Retrying connection...</span>
    `;
    await checkAPIConnection();
}

// ── Init ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initLightbox();
    initToasts();
    checkAPIConnection();
    initSkeletons(); 
});