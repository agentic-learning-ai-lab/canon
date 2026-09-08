window.HELP_IMPROVE_VIDEOJS = false;

// Manual per-task rollout carousels: one demo at a time, prev/next arrows.
// Runs independently of jQuery / bulma so the arrows always work even if those
// libraries fail to load or throw.
function initManualCarousels() {
    document.querySelectorAll('.manual-carousel').forEach(function (mc) {
        var slides = mc.querySelectorAll('.mc-slide');
        if (!slides.length) return;
        var counter = mc.querySelector('.mc-counter');
        var idx = 0;
        function show(n) {
            idx = (n + slides.length) % slides.length;
            slides.forEach(function (s, i) {
                var active = (i === idx);
                s.classList.toggle('is-active', active);
                s.style.display = active ? 'block' : 'none';
                if (active) {
                    s.currentTime = 0;
                    var pr = s.play();
                    if (pr && pr.catch) pr.catch(function () {});
                } else {
                    s.pause();
                }
            });
            if (counter) counter.textContent = (idx + 1) + ' / ' + slides.length;
        }
        var prev = mc.querySelector('.mc-prev');
        var next = mc.querySelector('.mc-next');
        if (prev) prev.addEventListener('click', function () { show(idx - 1); });
        if (next) next.addEventListener('click', function () { show(idx + 1); });
        show(0);
    });
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initManualCarousels);
} else {
    initManualCarousels();
}

// Bulma carousel/slider init (best-effort; must not block the manual carousels above).
$(document).ready(function () {
    try {
        var options = {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            infinite: true,
            autoplay: true,
            autoplaySpeed: 5000,
        };
        if (window.bulmaCarousel) bulmaCarousel.attach('.carousel', options);
        if (window.bulmaSlider) bulmaSlider.attach();
    } catch (e) {
        /* no-op: manual carousels already initialized above */
    }
});
