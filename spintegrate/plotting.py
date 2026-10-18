"""Shared, GUI-independent spectrum plots for live analysis and saved-fit review."""
from .appearance import area_color
from .i18n import tr


def draw_spectrum(overview, detail, residual, spectrum, session, selected_region,
                  analysis=None, local=None, component_limit=60, language=None, show_region_names=True):
    language = language or session.appearance.language
    t = lambda text: tr(text, language)
    for ax, title in ((overview, 'Весь спектр'), (detail, 'Выбранная область'),
                      (residual, 'Остаток: данные − подгонка')):
        ax.clear()
        ax.set_title(t(title), fontsize=10, loc='left')
        ax.set_xlabel(t('Химический сдвиг (ppm)'))
        ax.grid(alpha=.15)
    overview.set_ylabel(t('Интенсивность (отн. ед.)'))
    detail.set_ylabel(t('Интенсивность (отн. ед.)'))
    if spectrum is None:
        return
    overview.plot(spectrum.ppm, spectrum.intensity, color='#254b79', lw=.8)
    overview.set_xlim(spectrum.ppm[-1], spectrum.ppm[0])
    for r in session.regions:
        overview.axvspan(r.low, r.high, color='#4d8bbb', alpha=.17)
        if show_region_names:
            overview.text((r.low+r.high)/2, .95, r.name, transform=overview.get_xaxis_transform(),
                          ha='center', va='top', fontsize=8)
    for index, r in enumerate(session.integrals):
        overview.axvspan(r.low, r.high, color=area_color(r, index), alpha=.23)
    result = analysis.results.get(selected_region) if analysis else None
    region = next((r for r in session.regions if r.name == selected_region), None)
    if result:
        x, y = result.x, result.corrected
        detail.plot(x, y, color='#254b79', lw=1., label=t('После базовой линии'))
        detail.plot(x, result.fitted, color='#d26335', lw=1.4, label=t('Сумма Лоренцев'))
        components = result.components()
        flagged = {p[k] for p in result.diagnostics.get('cancellation_pairs', []) for k in ('peak1', 'peak2')}
        count = len(result.centers) if component_limit is None else min(len(result.centers), component_limit)
        for i in range(count):
            detail.plot(x, components[:, i], '--', lw=1.2 if i+1 in flagged else .75,
                        alpha=.9 if i+1 in flagged else .6)
            if i+1 in flagged:
                detail.annotate(f'L{i+1}', (result.centers[i], result.amplitudes[i]), fontsize=8)
        detail.scatter(result.centers, result.initial_amplitudes, color='#c33b43',
                       s=20, zorder=4, label=t('Peak-picking (фикс. центры)'))
        residual.plot(x, y-result.fitted, color='#687689', lw=.85)
        title = (f'{result.region}: ' if show_region_names else '') + f'{len(result.centers)} пиков; '
        title += 'сошлось' if result.success else 'НЕ СОШЛОСЬ'
        detail.set_title(t(title), fontsize=10, loc='left')
    elif local:
        x, raw, y, line, peaks, _ = local
        detail.plot(x, raw, color='#b0b9c5', lw=.9, label=t('До базовой линии'))
        detail.plot(x, line, color='#d26335', lw=1., label=t('Базовая линия'))
        detail.plot(x, y, color='#254b79', lw=1., label=t('После коррекции'))
        detail.scatter(*peaks, color='#c33b43', s=22, zorder=4, label='Peak-picking')
        detail.axhline(session.fit.threshold, color='#c33b43', ls=':', lw=.8, label='Threshold')
        if session.fit.signed:
            detail.axhline(-session.fit.threshold, color='#c33b43', ls=':', lw=.8)
    elif region:
        mask = (spectrum.ppm >= region.low) & (spectrum.ppm <= region.high)
        detail.plot(spectrum.ppm[mask], spectrum.intensity[mask], color='#254b79', lw=.8, label=t('Спектр'))
        if analysis and region.name in analysis.errors:
            detail.text(.02, .92, ('Fit error: ' if language == 'en' else 'Ошибка подгонки: ') + t(analysis.errors[region.name]),
                        transform=detail.transAxes, va='top', fontsize=8, wrap=True, color='#ad3434')
    else:
        detail.plot(spectrum.ppm, spectrum.intensity, color='#254b79', lw=.8, label=t('Спектр'))
    if region:
        detail.set_xlim(region.high, region.low)
        for index, area in enumerate(session.integrals):
            if area.parent == region.name:
                detail.axvspan(area.low, area.high, color=area_color(area, index), alpha=.20)
    else:
        detail.set_xlim(spectrum.ppm[-1], spectrum.ppm[0])
    detail.legend(fontsize=8, loc='upper right')
    residual.axhline(0, color='#a0a8b5', lw=.7)
