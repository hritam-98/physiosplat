/* ============ PhysioSplat site interactions ============ */
(function () {
  "use strict";

  /* ---------- Animated "gaussian splat" background ---------- */
  const canvas = document.getElementById("splat-bg");
  const ctx = canvas.getContext("2d");
  let W, H, blobs, raf;
  const PALETTE = [
    [76, 201, 240],   // cyan
    [247, 37, 133],   // magenta
    [255, 209, 102],  // gold
    [123, 155, 255],  // periwinkle
  ];
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    W = canvas.width = window.innerWidth * devicePixelRatio;
    H = canvas.height = window.innerHeight * devicePixelRatio;
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";
  }

  function makeBlobs() {
    const n = Math.min(46, Math.floor((window.innerWidth * window.innerHeight) / 26000));
    blobs = Array.from({ length: n }, () => {
      const c = PALETTE[(Math.random() * PALETTE.length) | 0];
      return {
        x: Math.random() * W,
        y: Math.random() * H,
        // anisotropic gaussian -> oriented ellipse (mimics a splat)
        rx: (40 + Math.random() * 150) * devicePixelRatio,
        ry: (24 + Math.random() * 90) * devicePixelRatio,
        rot: Math.random() * Math.PI,
        vx: (Math.random() - 0.5) * 0.22 * devicePixelRatio,
        vy: (Math.random() - 0.5) * 0.22 * devicePixelRatio,
        vr: (Math.random() - 0.5) * 0.0016,
        a: 0.05 + Math.random() * 0.12,
        c,
      };
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    for (const b of blobs) {
      b.x += b.vx; b.y += b.vy; b.rot += b.vr;
      if (b.x < -300) b.x = W + 300; if (b.x > W + 300) b.x = -300;
      if (b.y < -300) b.y = H + 300; if (b.y > H + 300) b.y = -300;
      ctx.save();
      ctx.translate(b.x, b.y);
      ctx.rotate(b.rot);
      ctx.scale(b.rx, b.ry);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
      const [r, gg, bb] = b.c;
      g.addColorStop(0, `rgba(${r},${gg},${bb},${b.a})`);
      g.addColorStop(1, `rgba(${r},${gg},${bb},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    raf = requestAnimationFrame(draw);
  }

  if (canvas) {
    resize(); makeBlobs();
    if (!reduce) draw(); else { /* draw one static frame */ draw(); cancelAnimationFrame(raf); }
    let to;
    window.addEventListener("resize", () => {
      clearTimeout(to);
      to = setTimeout(() => { resize(); makeBlobs(); }, 200);
    });
  }

  /* ---------- Scroll reveal ---------- */
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }),
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ---------- Count-up hero stats ---------- */
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    if (isNaN(target)) return;
    const dec = parseInt(el.dataset.dec || "0", 10);
    const suffix = el.dataset.suffix || "";
    const dur = 1400; const t0 = performance.now();
    function step(t) {
      const p = Math.min((t - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  const statIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { countUp(e.target); statIO.unobserve(e.target); } });
  }, { threshold: 0.6 });
  document.querySelectorAll(".hstat b[data-count]").forEach((el) => statIO.observe(el));

  /* ---------- Animated metric bars ---------- */
  const barIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.querySelectorAll(".bar-ours").forEach((b, i) => {
          setTimeout(() => { b.style.setProperty("--w", b.dataset.w); b.classList.add("show"); }, 120 * i);
        });
        e.target.querySelectorAll(".bar-prev").forEach((b) => b.classList.add("show"));
        barIO.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });
  document.querySelectorAll(".bars").forEach((el) => barIO.observe(el));

  /* ---------- Drag-to-compare slider ---------- */
  const widget = document.getElementById("compare-widget");
  if (widget) {
    const front = document.getElementById("cmp-front");
    const handle = document.getElementById("cmp-handle");
    let dragging = false;
    function setPos(clientX) {
      const r = widget.getBoundingClientRect();
      let p = ((clientX - r.left) / r.width) * 100;
      p = Math.max(2, Math.min(98, p));
      front.style.width = p + "%";
      handle.style.left = p + "%";
    }
    const start = () => (dragging = true);
    const end = () => (dragging = false);
    const move = (x) => { if (dragging) setPos(x); };

    widget.addEventListener("mousedown", (e) => { start(); setPos(e.clientX); });
    window.addEventListener("mousemove", (e) => move(e.clientX));
    window.addEventListener("mouseup", end);
    widget.addEventListener("touchstart", (e) => { start(); setPos(e.touches[0].clientX); }, { passive: true });
    window.addEventListener("touchmove", (e) => move(e.touches[0].clientX), { passive: true });
    window.addEventListener("touchend", end);

    // subtle auto-demo on first reveal
    const demoIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          let p = 50, dir = 1, n = 0;
          const id = setInterval(() => {
            p += dir * 1.4; if (p > 72 || p < 30) dir *= -1;
            front.style.width = p + "%"; handle.style.left = p + "%";
            if (++n > 60) { clearInterval(id); front.style.width = "50%"; handle.style.left = "50%"; }
          }, 16);
          demoIO.unobserve(e.target);
        }
      });
    }, { threshold: 0.5 });
    demoIO.observe(widget);
  }

  /* ---------- Copy buttons ---------- */
  function wireCopy(btnId, srcId) {
    const btn = document.getElementById(btnId);
    const src = document.getElementById(srcId);
    if (!btn || !src) return;
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(src.innerText.trim()).then(() => {
        const old = btn.textContent;
        btn.textContent = "copied ✓"; btn.classList.add("done");
        setTimeout(() => { btn.textContent = old; btn.classList.remove("done"); }, 1600);
      });
    });
  }
  wireCopy("copy-btn", "code-block");
  wireCopy("copy-bib", "bib-block");

  /* ---------- Nav scroll state ---------- */
  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 40);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
})();
