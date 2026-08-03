/* ==========================================================================
   OpenSees-MCP — Landing motion v2
   - Scroll-triggered reveal (IntersectionObserver)
   - Hero title word stagger
   - Counter animation
   - Nav background switch on scroll
   - Smooth anchor scroll
   - Strong scroll parallax engine (data-parallax)
   - Mouse 3D tilt on cards
   - Strong hero mouse parallax with perspective
   - Hero scroll fade
   ========================================================================== */

(function () {
    'use strict';

    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ───── Nav scroll state ───── */
    const nav = document.getElementById('nav');
    if (nav) {
        const updateNav = () => {
            nav.dataset.state = window.scrollY > 8 ? 'scrolled' : 'top';
        };
        updateNav();
        window.addEventListener('scroll', updateNav, { passive: true });
    }

    /* ───── Hero title — trigger immediately ───── */
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        requestAnimationFrame(() => heroTitle.classList.add('is-visible'));
    }

    /* ───── Reveal observer ───── */
    if ('IntersectionObserver' in window && !prefersReduced) {
        const revealEls = document.querySelectorAll('.reveal, .reveal-fade, .reveal-scale');
        const io = new IntersectionObserver((entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    e.target.classList.add('is-visible');
                    io.unobserve(e.target);
                }
            });
        }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
        revealEls.forEach((el) => io.observe(el));
    } else {
        document.querySelectorAll('.reveal, .reveal-fade, .reveal-scale')
            .forEach((el) => el.classList.add('is-visible'));
    }

    /* ───── Counter animation ───── */
    const animateCount = (el) => {
        const target = parseFloat(el.dataset.target || '0');
        if (!isFinite(target) || target === 0) {
            el.textContent = '0';
            return;
        }
        const duration = 1400;
        const start = performance.now();
        const isInt = Number.isInteger(target);
        const step = (now) => {
            const t = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - t, 3);
            const value = eased * target;
            el.textContent = isInt ? Math.floor(value).toLocaleString() : value.toFixed(2);
            if (t < 1) requestAnimationFrame(step);
            else el.textContent = isInt ? target.toLocaleString() : target.toFixed(2);
        };
        requestAnimationFrame(step);
    };
    const counters = document.querySelectorAll('.count');
    if (counters.length) {
        if ('IntersectionObserver' in window && !prefersReduced) {
            const cIO = new IntersectionObserver((entries) => {
                entries.forEach((e) => {
                    if (e.isIntersecting) {
                        animateCount(e.target);
                        cIO.unobserve(e.target);
                    }
                });
            }, { threshold: 0.4 });
            counters.forEach((c) => cIO.observe(c));
        } else {
            counters.forEach((c) => {
                const target = parseFloat(c.dataset.target || '0');
                c.textContent = Number.isInteger(target) ? target.toLocaleString() : target.toFixed(2);
            });
        }
    }

    /* ───── Demo step auto-cycle ───── */
    const demoSteps = document.querySelectorAll('.demo-step');
    if (demoSteps.length && !prefersReduced) {
        let active = -1;
        demoSteps.forEach((s, i) => { if (s.classList.contains('demo-step--active')) active = i; });
        if (active < 0) active = 0;
        const cycle = () => {
            demoSteps[active].classList.remove('demo-step--active');
            active = (active + 1) % demoSteps.length;
            demoSteps[active].classList.add('demo-step--active');
        };
        const demo = document.getElementById('demo');
        if (demo && 'IntersectionObserver' in window) {
            const dIO = new IntersectionObserver((entries) => {
                entries.forEach((e) => {
                    if (e.isIntersecting && !demo.dataset.cycling) {
                        demo.dataset.cycling = '1';
                        setInterval(cycle, 2500);
                    }
                });
            }, { threshold: 0.3 });
            dIO.observe(demo);
        }
    }

    /* ───── Smooth anchor scroll with offset ───── */
    document.querySelectorAll('a[href^="#"]').forEach((a) => {
        a.addEventListener('click', (e) => {
            const id = a.getAttribute('href');
            if (!id || id === '#') return;
            const target = document.querySelector(id);
            if (!target) return;
            e.preventDefault();
            target.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth', block: 'start' });
            history.pushState(null, '', id);
        });
    });

    /* ───── SCROLL PARALLAX ENGINE ─────
       Reads data-parallax (translateY speed), data-parallax-x, data-parallax-rotate,
       data-parallax-scale, data-parallax-fade attributes. Computes from element's
       distance to viewport center for a "drift-through" effect.

       Speed convention: positive = lags behind (slower than scroll), negative = leads.
    */
    if (!prefersReduced) {
        const parallaxItems = [];
        document.querySelectorAll('[data-parallax], [data-parallax-x], [data-parallax-rotate], [data-parallax-scale], [data-parallax-fade]').forEach((el) => {
            parallaxItems.push({
                el,
                ty:    parseFloat(el.dataset.parallax || '0'),
                tx:    parseFloat(el.dataset.parallaxX || '0'),
                rot:   parseFloat(el.dataset.parallaxRotate || '0'),
                scl:   parseFloat(el.dataset.parallaxScale || '0'),
                fade:  parseFloat(el.dataset.parallaxFade || '0'),
            });
            el.style.willChange = 'transform, opacity';
        });

        let vh = window.innerHeight;
        let ticking = false;

        const update = () => {
            for (const it of parallaxItems) {
                const rect = it.el.getBoundingClientRect();
                if (rect.bottom < -400 || rect.top > vh + 400) continue;

                const elCenter = rect.top + rect.height / 2;
                const fromCenter = elCenter - vh / 2; // negative above center, positive below

                const dY = -fromCenter * it.ty;
                const dX = -fromCenter * it.tx;
                const dRot = -fromCenter * it.rot * 0.01;
                const dScl = 1 + (-fromCenter * it.scl * 0.0005);

                let transform = '';
                if (it.ty || it.tx) transform += `translate3d(${dX}px, ${dY}px, 0) `;
                if (it.rot)         transform += `rotate(${dRot}deg) `;
                if (it.scl)         transform += `scale(${dScl}) `;

                // Preserve any base transform from CSS by setting via style only if needed
                it.el.style.transform = transform.trim();

                if (it.fade) {
                    // Fade out as element scrolls past viewport (linear)
                    const progress = Math.min(1, Math.max(0, (vh - rect.top) / (vh * 0.7)));
                    const opacity = 1 - Math.min(1, Math.max(0, it.fade * (1 - progress)));
                    it.el.style.opacity = opacity.toFixed(3);
                }
            }
            ticking = false;
        };

        const onScroll = () => {
            if (!ticking) {
                requestAnimationFrame(update);
                ticking = true;
            }
        };

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', () => { vh = window.innerHeight; update(); }, { passive: true });
        update();
    }

    /* ───── HERO scroll fade (separate from parallax to keep word-stagger transform) ───── */
    const heroText = document.querySelector('.hero-text');
    const heroSection = document.querySelector('.hero');
    if (heroText && heroSection && !prefersReduced) {
        let raf = 0;
        const updateFade = () => {
            const rect = heroSection.getBoundingClientRect();
            const progress = Math.max(0, Math.min(1, -rect.top / (rect.height * 0.65)));
            heroText.style.opacity = (1 - progress * 0.9).toFixed(3);
            heroText.style.transform = `translate3d(0, ${-progress * 60}px, 0)`;
            raf = 0;
        };
        window.addEventListener('scroll', () => {
            if (!raf) raf = requestAnimationFrame(updateFade);
        }, { passive: true });
        updateFade();
    }

    /* ───── HERO mouse parallax — stronger, with perspective rotation ───── */
    const heroArt = document.querySelector('.hero-art .mock');
    if (heroArt && heroSection && !prefersReduced) {
        let raf = 0;
        let tx = 0, ty = 0;
        const apply = () => {
            heroArt.style.transform = `perspective(1100px) rotateY(${tx * 0.45}deg) rotateX(${-ty * 0.45}deg) translate3d(${tx}px, ${ty}px, 0)`;
            raf = 0;
        };
        heroSection.addEventListener('mousemove', (e) => {
            const rect = heroSection.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            tx = x * 26;
            ty = y * 18;
            if (!raf) raf = requestAnimationFrame(apply);
        });
        heroSection.addEventListener('mouseleave', () => {
            tx = 0; ty = 0;
            heroArt.style.transition = 'transform .7s cubic-bezier(.16,1,.3,1)';
            heroArt.style.transform = 'perspective(1100px) rotateY(0) rotateX(0) translate3d(0,0,0)';
            setTimeout(() => { heroArt.style.transition = ''; }, 700);
        });
    }

    /* ───── 3D card tilt on mouse over .tilt ───── */
    if (!prefersReduced) {
        document.querySelectorAll('.tilt').forEach((el) => {
            let raf = 0;
            let rx = 0, ry = 0, ts = 1;
            const apply = () => {
                el.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) scale(${ts}) translateY(-4px)`;
                raf = 0;
            };
            el.addEventListener('mousemove', (e) => {
                const rect = el.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                ry = x * 8;
                rx = -y * 6;
                ts = 1.02;
                if (!raf) raf = requestAnimationFrame(apply);
            });
            el.addEventListener('mouseleave', () => {
                rx = 0; ry = 0; ts = 1;
                el.style.transition = 'transform .5s cubic-bezier(.16,1,.3,1)';
                el.style.transform = 'perspective(900px) rotateX(0) rotateY(0) scale(1) translateY(0)';
                setTimeout(() => { el.style.transition = ''; }, 500);
            });
        });
    }
})();
