import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO

# Page config
st.set_page_config(page_title="Nova — Modern SaaS UI", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

# ---------- Helper: inline SVG/logo ----------
LOGO_SVG = """
<svg width="46" height="46" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="48" height="48" rx="12" fill="url(#g)"/>
  <path d="M14 32 L24 16 L34 32 Z" fill="white" opacity="0.95"/>
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6366F1"/>
      <stop offset="1" stop-color="#A78BFA"/>
    </linearGradient>
  </defs>
</svg>
"""

HERO_SVG = """
<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
  <defs>
    <linearGradient id="lg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#7c3aed" stop-opacity="0.95"/>
      <stop offset="1" stop-color="#06b6d4" stop-opacity="0.9"/>
    </linearGradient>
    <filter id="f" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="30" result="b"/>
      <feBlend in="SourceGraphic" in2="b"/>
    </filter>
  </defs>
  <g filter="url(#f)">
    <circle cx="150" cy="140" r="130" fill="url(#lg)"/>
    <rect x="360" y="80" width="300" height="220" rx="20" fill="white" opacity="0.04"/>
  </g>
  <g transform="translate(200,120)">
    <rect x="0" y="0" width="320" height="200" rx="18" fill="white" opacity="0.06"/>
    <g transform="translate(18,18)" fill="white" opacity="0.95">
      <rect x="0" y="0" width="120" height="16" rx="8"/>
      <rect x="0" y="36" width="280" height="12" rx="6"/>
      <rect x="0" y="64" width="240" height="12" rx="6"/>
      <circle cx="260" cy="40" r="20" fill="#34D399" opacity="0.95"/>
    </g>
  </g>
</svg>
"""

# ---------- CSS (injected) ----------
CSS = """
<style>
:root{
  --bg:#0f172a;
  --card:#0b1220;
  --muted: #94a3b8;
  --accent1: #7c3aed;
  --accent2: #06b6d4;
  --glass: rgba(255,255,255,0.03);
}
body {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,0.12), transparent 10%),
              radial-gradient(1000px 500px at 90% 90%, rgba(6,182,212,0.08), transparent 10%),
              var(--bg) !important;
  color: #E6EEF3;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
header .css-1d391kg { padding: 0 } /* keep Streamlit top padding tidy */
.hero {
  padding: 48px 36px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  margin-bottom: 24px;
}
.btn-cta {
  display:inline-block;
  padding: 12px 22px;
  border-radius: 12px;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  color: white;
  font-weight: 600;
  text-decoration: none;
  box-shadow: 0 6px 20px rgba(12,22,70,0.45);
}
.feature-card {
  background: var(--card);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 6px 24px rgba(2,6,23,0.5);
}
.small-muted { color: var(--muted); font-size:14px; }
.pulse {
  animation: pulse 2.8s infinite;
}
@keyframes pulse {
  0% { transform: translateY(0); opacity: 1; }
  50% { transform: translateY(-6px); opacity: .9; }
  100% { transform: translateY(0); opacity: 1; }
}

/* Responsive tweaks */
@media (max-width: 900px) {
  .hero { padding: 26px; }
  .btn-cta { padding: 10px 16px; }
}
</style>
"""

# Inject CSS
components.html(CSS, height=0)

# ---------- Top nav ----------
with st.container():
    cols = st.columns([1,6,1])
    with cols[0]:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div style='display:flex;gap:18px;align-items:center;justify-content:center;'>"
                    "<a class='small-muted' href='#features'>Features</a>"
                    "<a class='small-muted' href='#pricing'>Pricing</a>"
                    "<a class='small-muted' href='#testimonials'>Testimonials</a>"
                    "<a class='small-muted' href='#contact'>Contact</a>"
                    "</div>", unsafe_allow_html=True)
    with cols[2]:
        st.write("")  # reserved for action
        st.markdown("<div style='text-align:right'><a class='btn-cta' href='#contact'>Start free trial</a></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Hero ----------
def render_hero():
    left, right = st.columns([5,5])
    with left:
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.markdown("<h1 style='margin:0 0 6px 0;font-size:44px;line-height:1.03'>Build delightful products — faster.</h1>", unsafe_allow_html=True)
        st.markdown("<p class='small-muted' style='margin-top:6px;font-size:16px'>Nova helps teams launch and iterate on web products with intuitive UI components, analytics and production-ready integrations.</p>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px;display:flex;gap:12px;align-items:center'>"
                    "<a class='btn-cta' href='#contact'>Start free — no credit card</a>"
                    "<a class='small-muted' href='#pricing' style='padding:10px 14px;background:transparent;border-radius:10px;text-decoration:none;'>See pricing →</a>"
                    "</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:20px' class='small-muted'>Trusted by teams at <strong>Leaf</strong>, <strong>Rover</strong> and <strong>Nova</strong>.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        # show inline hero svg inside a nice card
        st.markdown("<div class='feature-card' style='display:flex;align-items:center;justify-content:center;height:320px'>"
                    f"{HERO_SVG}"
                    "</div>", unsafe_allow_html=True)

render_hero()

st.markdown("<h2 id='features'>Features</h2>", unsafe_allow_html=True)

# ---------- Features ----------
def render_features():
    cols = st.columns(3, gap="large")
    feature_list = [
        ("Pixel-perfect UI", "Ship beautiful pages and components, with production-friendly defaults."),
        ("Integrations", "Connect databases, auth providers, and analytics in minutes."),
        ("Secure by default", "Role-based access, encryption at rest, and audit logs."),
    ]
    for col, feat in zip(cols, feature_list):
        with col:
            st.markdown(f"<div class='feature-card'>"
                        f"<h3 style='margin:0 0 6px 0'>{feat[0]}</h3>"
                        f"<p class='small-muted' style='margin:0'>{feat[1]}</p>"
                        f"</div>", unsafe_allow_html=True)
render_features()

st.markdown("---")

# ---------- Stats strip ----------
with st.container():
    a, b, c, d = st.columns(4)
    a.markdown("<div class='feature-card' style='text-align:center'><h2 style='margin:2px'>2M+</h2><div class='small-muted'>Users onboarded</div></div>", unsafe_allow_html=True)
    b.markdown("<div class='feature-card' style='text-align:center'><h2 style='margin:2px'>99.99%</h2><div class='small-muted'>Uptime SLA</div></div>", unsafe_allow_html=True)
    c.markdown("<div class='feature-card' style='text-align:center'><h2 style='margin:2px'>1m</h2><div class='small-muted'>Avg time to deploy</div></div>", unsafe_allow_html=True)
    d.markdown("<div class='feature-card' style='text-align:center'><h2 style='margin:2px'>300+</h2><div class='small-muted'>Integrations</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Pricing ----------
st.markdown("<h2 id='pricing'>Pricing</h2>", unsafe_allow_html=True)

with st.container():
    p1, p2, p3 = st.columns([1,1,1], gap="large")
    with p1:
        st.markdown("<div class='feature-card' style='text-align:left'>"
                    "<h3 style='margin:0'>Starter</h3>"
                    "<p class='small-muted' style='margin:0'>For small teams & experiments</p>"
                    "<h2 style='margin-top:10px'>$0<span style='font-size:14px;color:var(--muted)'>/mo</span></h2>"
                    "<ul class='small-muted'><li>Up to 5 teammates</li><li>Core components</li><li>Email support</li></ul>"
                    "<button class='btn-cta' onclick='window.scrollTo(0, document.body.scrollHeight)'>Get started</button>"
                    "</div>", unsafe_allow_html=True)
    with p2:
        # highlighted plan
        st.markdown("<div class='feature-card' style='border: 1px solid rgba(255,255,255,0.04);text-align:left'>"
                    "<div style='display:flex;justify-content:space-between;align-items:center'><h3 style='margin:0'>Pro</h3><div style='background:linear-gradient(90deg,var(--accent1),var(--accent2));padding:6px 10px;border-radius:8px;font-weight:600'>Most popular</div></div>"
                    "<p class='small-muted' style='margin:0'>Power teams</p>"
                    "<h2 style='margin-top:10px'>$29<span style='font-size:14px;color:var(--muted)'>/mo</span></h2>"
                    "<ul class='small-muted'><li>Unlimited projects</li><li>Priority support</li><li>SSO & SAML</li></ul>"
                    "<button class='btn-cta' onclick='window.scrollTo(0, document.body.scrollHeight)'>Buy pro</button>"
                    "</div>", unsafe_allow_html=True)
    with p3:
        st.markdown("<div class='feature-card' style='text-align:left'>"
                    "<h3 style='margin:0'>Enterprise</h3>"
                    "<p class='small-muted' style='margin:0'>Custom plans & SLAs</p>"
                    "<h2 style='margin-top:10px'>Contact us</h2>"
                    "<ul class='small-muted'><li>Custom integrations</li><li>Monthly billing</li><li>Dedicated support</li></ul>"
                    "<button class='btn-cta' onclick='window.scrollTo(0, document.body.scrollHeight)'>Contact sales</button>"
                    "</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Testimonials ----------
st.markdown("<h2 id='testimonials'>Testimonials</h2>", unsafe_allow_html=True)

with st.container():
    t1, t2, t3 = st.columns(3)
    test_data = [
        ("Aisha, Head of Product", "Nova cut our time to market in half. The UI components are delightful and consistent."),
        ("Liam, CTO", "Solid infra and easy SSO integration. Observability built-in — ship with confidence."),
        ("Priya, Lead Designer", "Design tokens + components matched our brand. Developers 🧡 designers."),
    ]
    for col, t in zip((t1, t2, t3), test_data):
        col.markdown(f"<div class='feature-card'><strong>{t[0]}</strong><p class='small-muted' style='margin:8px 0 0'>{t[1]}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Contact / CTA ----------
st.markdown("<h2 id='contact'>Contact</h2>", unsafe_allow_html=True)

with st.container():
    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.write("Tell us about your team — we’ll send you a trial and a quick setup guide.")
        with st.form("contact_form"):
            name = st.text_input("Full name", placeholder="Jane Doe")
            email = st.text_input("Work email", placeholder="you@company.com")
            org = st.text_input("Company (optional)")
            plan = st.selectbox("I'm interested in", ["Starter (Free)", "Pro", "Enterprise"])
            notes = st.text_area("How can we help?", placeholder="Short note (optional)")
            submitted = st.form_submit_button("Request access")
            if submitted:
                # In real app: store to DB or send email. Here we show a pretty success.
                st.success("Thanks! We’ll email you a trial link and setup tips.")
                st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='feature-card' style='text-align:center'>"
                    "<h4 style='margin-bottom:6px'>Need a demo?</h4>"
                    "<p class='small-muted'>Schedule a 20-minute walkthrough with our team.</p>"
                    "<a class='btn-cta' href='mailto:sales@nova.example?subject=Schedule%20demo'>Book demo</a>"
                    "</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Footer ----------
with st.container():
    f1, f2 = st.columns([3,1])
    with f1:
        st.markdown("<div style='display:flex;align-items:center;gap:12px'>"
                    f"{LOGO_SVG}"
                    "<div><strong>Nova</strong><div class='small-muted'>Design & build modern web products</div></div>"
                    "</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px' class='small-muted'>© 2025 Nova, Inc. All rights reserved.</div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div style='text-align:right' class='small-muted'>Privacy · Terms · Status</div>", unsafe_allow_html=True)

# ---------- Optional: small script to smooth-scroll anchors (works inside components.html) ----------
SCROLL_JS = """
<script>
document.querySelectorAll("a[href^='#']").forEach(a=>{
  a.addEventListener('click', e=>{
    e.preventDefault();
    let id = a.getAttribute('href').slice(1);
    let el = document.getElementById(id);
    if(el) {
      el.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  });
});
</script>
"""
components.html(SCROLL_JS, height=0)

# End





