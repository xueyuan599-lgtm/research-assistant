/**
 * Idea2Top — 交互主脚本
 * 语言切换、滚动动画、架构图交互、移动端导航
 */

(function () {
  'use strict';

  let currentLang = 'zh';

  // =========================================================
  // 1. i18n Render Engine
  // =========================================================
  function t(path) {
    const keys = path.split('.');
    let val = I18N_DATA[currentLang];
    for (const k of keys) {
      if (val === undefined || val === null) return path;
      val = val[k];
    }
    return val ?? path;
  }

  function renderAll() {
    // data-i18n attributes: set textContent
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const val = t(el.getAttribute('data-i18n'));
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        el.placeholder = val;
      } else {
        el.textContent = val;
      }
    });

    // data-i18n-html: set innerHTML
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });

    // data-i18n-flag: set attribute like href, src
    document.querySelectorAll('[data-i18n-flag]').forEach(el => {
      const attr = el.getAttribute('data-i18n-flag');
      const val = t(el.getAttribute('data-i18n-' + attr));
      if (val) el.setAttribute(attr, val);
    });

    // Update document lang
    document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';

    // Update lang toggle buttons
    document.querySelectorAll('.lang-option').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === currentLang);
    });

    // Re-render dynamic content
    renderWorkflowSteps();
    renderAgentCards();
    renderKnowledgeCategories();
    renderGuideSteps();
    renderStats();
  }

  // =========================================================
  // 2. Language Toggle
  // =========================================================
  function setLang(lang) {
    if (lang === currentLang) return;
    currentLang = lang;
    renderAll();
    localStorage.setItem('idea2top-lang', lang);
  }

  // =========================================================
  // 3. Dynamic Section Renderers
  // =========================================================

  // ---- Stats ----
  function renderStats() {
    const container = document.getElementById('stats-container');
    if (!container) return;
    const stats = I18N_DATA[currentLang].about.stats;
    container.innerHTML = Object.values(stats).map(s => `
      <div class="stat-card reveal">
        <span class="stat-number">${s.num}</span>
        <span class="stat-label">${s.label}<span class="en">${s.labelEn}</span></span>
      </div>
    `).join('');
  }

  // ---- Workflow Steps ----
  function renderWorkflowSteps() {
    const container = document.getElementById('workflow-steps');
    if (!container) return;
    const steps = I18N_DATA[currentLang].workflow.steps;
    container.innerHTML = steps.map((step, i) => `
      <div class="workflow-step reveal">
        <div class="step-dot"></div>
        <div class="step-num">STEP ${String(i + 1).padStart(2, '0')}</div>
        <h3>${step.title} <span class="en">${step.titleEn}</span></h3>
        <p>${step.desc}</p>
        <div class="step-agents">
          ${step.agents.map(a => `<span class="step-agent-tag">${a}</span>`).join('')}
        </div>
      </div>
    `).join('');
  }

  // ---- Agent Cards ----
  function renderAgentCards() {
    const container = document.getElementById('agent-cards');
    if (!container) return;
    const agents = I18N_DATA[currentLang].agents.list;
    container.innerHTML = agents.map(a => `
      <div class="agent-card reveal">
        <span class="card-icon">${a.icon}</span>
        <h3>${a.name}</h3>
        <span class="card-zh">${a.zh}</span>
        <p class="card-desc">${a.desc}</p>
        ${a.tags ? `
          <div class="card-tags">
            ${a.tags.map(t => `<span class="tag">${t}</span>`).join('')}
          </div>
        ` : ''}
      </div>
    `).join('');
  }

  // ---- Knowledge Categories ----
  function renderKnowledgeCategories() {
    const container = document.getElementById('knowledge-categories');
    if (!container) return;
    const cats = I18N_DATA[currentLang].knowledge.categories;
    container.innerHTML = cats.map(c => `
      <div class="knowledge-category reveal">
        <h4>${c.name}</h4>
        <span class="cat-zh">${c.nameEn}</span>
        <ul>
          ${c.items.map(item => `<li>${item}</li>`).join('')}
        </ul>
      </div>
    `).join('');
  }

  // ---- Guide Steps ----
  function renderGuideSteps() {
    const container = document.getElementById('guide-steps');
    if (!container) return;
    const steps = I18N_DATA[currentLang].guide.steps;
    container.innerHTML = steps.map((step, i) => `
      <div class="guide-step reveal">
        <div class="step-marker">${i + 1}</div>
        <div class="step-content">
          <h4>${step.title} <span class="en" style="font-weight:400;color:var(--ink-light);font-size:0.85rem">${step.titleEn}</span></h4>
          <p>${step.desc}</p>
          <code class="code-hint">${step.hint}</code>
        </div>
      </div>
    `).join('');
  }

  // =========================================================
  // 4. Architecture Diagram Interactivity
  // =========================================================
  function setupArchCards() {
    document.querySelectorAll('.arch-card').forEach(card => {
      card.addEventListener('click', function (e) {
        // Don't toggle if clicking a link inside
        if (e.target.closest('a')) return;
        this.classList.toggle('expanded');
      });
    });
  }

  // =========================================================
  // 5. Scroll-Based Reveal Animations
  // =========================================================
  function setupScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          // Don't unobserve so re-renders re-trigger
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    // Observe all .reveal elements (including newly rendered ones)
    function observeReveals() {
      document.querySelectorAll('.reveal').forEach(el => {
        if (!el.classList.contains('visible')) {
          observer.observe(el);
        }
      });
    }

    // Initial observe
    observeReveals();

    // Re-observe after renders
    const renderObserver = new MutationObserver(() => observeReveals());
    const mainContent = document.querySelector('main');
    if (mainContent) {
      renderObserver.observe(mainContent, { childList: true, subtree: true });
    }
  }

  // =========================================================
  // 6. Smooth Scroll for Nav Links
  // =========================================================
  function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });

          // Close mobile nav
          document.querySelector('.nav-links')?.classList.remove('open');
        }
      });
    });
  }

  // =========================================================
  // 7. Navbar scroll effect
  // =========================================================
  function setupNavScroll() {
    const nav = document.querySelector('.nav');
    let ticking = false;

    window.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          nav.classList.toggle('scrolled', window.scrollY > 20);
          ticking = false;
        });
        ticking = true;
      }
    });
  }

  // =========================================================
  // 8. Mobile Nav Toggle
  // =========================================================
  function setupMobileNav() {
    const toggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (toggle && navLinks) {
      toggle.addEventListener('click', () => {
        navLinks.classList.toggle('open');
      });

      // Close on outside click
      document.addEventListener('click', (e) => {
        if (!toggle.contains(e.target) && !navLinks.contains(e.target)) {
          navLinks.classList.remove('open');
        }
      });
    }
  }

  // =========================================================
  // 9. Hero scroll indicator click
  // =========================================================
  function setupScrollIndicator() {
    const indicator = document.querySelector('.scroll-indicator');
    if (indicator) {
      indicator.addEventListener('click', () => {
        const about = document.getElementById('about');
        if (about) about.scrollIntoView({ behavior: 'smooth' });
      });
      indicator.style.cursor = 'pointer';
    }
  }

  // =========================================================
  // 10. Initialize
  // =========================================================
  function init() {
    // Restore saved language
    const saved = localStorage.getItem('idea2top-lang');
    if (saved && (saved === 'zh' || saved === 'en')) {
      currentLang = saved;
    }

    // Set up language toggle listeners
    document.querySelectorAll('.lang-option').forEach(btn => {
      btn.addEventListener('click', () => setLang(btn.dataset.lang));
    });

    // Initial render
    renderAll();

    // Set up interactivity
    setupArchCards();
    setupScrollReveal();
    setupSmoothScroll();
    setupNavScroll();
    setupMobileNav();
    setupScrollIndicator();
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
