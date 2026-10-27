"""UI translations. Region names, file paths and numeric settings remain user data."""
import re
import string
import tkinter as tk
from tkinter import ttk


EN = {
    'Spintegrate — анализ спектров ЯМР (raw)': 'Spintegrate — raw NMR analysis',
    'Spintegrate — анализ из pdata': 'Spintegrate — processed pdata analysis',
    'Анализировать pdata…': 'Analyze pdata…', 'Анализировать raw…': 'Analyze raw…',
    'Открыть pdata…': 'Open pdata…', 'Номер pdata (procno)': 'pdata processing number',
    'Метод анализа': 'Analysis method', 'Деконволюция Лоренцами': 'Lorentzian deconvolution',
    'Весь спектр + область + интеграл': 'Full spectrum + region + integral', 'Область + интеграл': 'Region + integral',
    'Неизвестный метод анализа: lorentzian или direct.': 'Analysis method must be lorentzian or direct.',
    'Источник спектра должен быть raw или pdata.': 'Spectrum source must be raw or pdata.',
    'Прямое интегрирование областей': 'Direct region integration',
    'Прямое интегрирование': 'Direct integration',
    'Накопленный интеграл по областям': 'Cumulative region integrals',
    'Интеграл (инт. × ppm)': 'Integral (intensity × ppm)',
    'По умолчанию используются 1r и ppm-шкала TopSpin. Фазы PHC0/PHC1 повторно не применяются. Выбранные здесь фаза, LB и zero filling — дополнительная обработка; для неё нужен канал 1i.':
        'By default, TopSpin 1r and its ppm scale are used. PHC0/PHC1 are not applied again. Phase, LB and zero filling here are additional processing and require 1i.',
    'Прямой интеграл = ∫ I(ppm) dppm после локальной коррекции базовой линии. Поиск пиков и подгонка не выполняются. Отрицательные сигналы дают отрицательный вклад.':
        'Direct integral = ∫ I(ppm) dppm after local baseline correction. No peak picking or optimization is used. Negative signals contribute negative area.',
    'Папка pdata с 1r и procs или папка эксперимента': 'pdata folder with 1r and procs, or experiment folder',
    'Загрузка pdata…': 'Loading processed pdata…', 'Интегрирование областей…': 'Integrating regions…',
    'Создана синтетическая серия из четырёх pdata.': 'Created four synthetic pdata spectra.',
    'Сохранить интегралы CSV…': 'Export integrals CSV…',
    'Единицы: интенсивность × ppm; численный интеграл по точным границам.':
        'Units: intensity × ppm; numerical integration with exact bounds.',
    'Прямое интегрирование: поиск пиков и оптимизация не используются.': 'Direct integration: no peak picking or optimization.',
    'Для дополнительной аподизации, фазировки, abs или zero filling pdata нужен файл 1i. Без 1i доступны исходная фазировка TopSpin, базовая линия и анализ областей.':
        'Additional pdata apodization, phase correction, abs or zero filling requires 1i. Without it, imported TopSpin phasing, local baseline correction and region analysis remain available.',
    'Дополнительная обработка применена к спектру после TopSpin; исходная фазировка не повторяется.':
        'Additional processing was applied after TopSpin; imported phase correction was not repeated.',
    'Номер обработки pdata должен быть положительным целым числом.': 'pdata processing number must be a positive integer.',
    'Для pdata выберите файл 1r или папку с 1r и procs.': 'For pdata, select 1r or a folder containing 1r and procs.',
    'Поддерживаются только одномерные обработанные спектры pdata.': 'Only one-dimensional processed pdata spectra are supported.',
    'В procs нужны корректные SF, SW_p, OFFSET и SI для шкалы TopSpin.': 'Valid SF, SW_p, OFFSET and SI in procs are required for the TopSpin axis.',
    'Некорректная шкала TopSpin: SF, SW_p, OFFSET или SI.': 'Invalid TopSpin axis: SF, SW_p, OFFSET or SI.',
    'Файлы 1r/1i короче размера, указанного в procs.': '1r/1i are shorter than the size specified in procs.',
    'В обработанном спектре есть нечисловые значения.': 'Processed spectrum contains nonfinite values.',
    'Некорректные PHC0/PHC1 в procs.': 'Invalid PHC0/PHC1 in procs.',
    'Обрезанные pdata (STSR/STSI/FTSIZE) пока не поддерживаются: сохраните полный спектр в TopSpin или откройте raw, чтобы не исказить шкалу ppm.':
        'Cropped pdata (STSR/STSI/FTSIZE) is not supported yet. Save a full spectrum in TopSpin or open raw to preserve the ppm axis.',
    'Для интегрирования нужны конечные одномерные массивы одинаковой длины.': 'Integration requires finite 1D arrays of equal length.',
    'Шкала интегрирования должна быть строго монотонной без повторов.': 'Integration axis must be strictly monotonic without duplicates.',
    'Для интегрирования нужны две различные границы low < high.': 'Integration requires two distinct bounds low < high.',
    'Область интегрирования выходит за диапазон спектра.': 'Integration region is outside the spectrum.',
    'Spintegrate — анализ спектров ЯМР': 'Spintegrate — NMR spectrum analysis',
    'Загрузить настройки JSON': 'Load JSON settings', 'Сохранить настройки JSON': 'Save JSON settings',
    'Один спектр': 'Single spectrum', 'Много спектров': 'Batch analysis',
    'Открыть Bruker…': 'Open Bruker…', 'Открыть NPZ…': 'Open NPZ…', 'Демо': 'Demo',
    'Аподизация / базовая линия…': 'Apodization / baseline…', 'Выполнить анализ': 'Run analysis',
    'Запомнить → серия': 'Use for batch →', 'Спектр не загружен': 'No spectrum loaded',
    'Обработка': 'Processing', 'Области': 'Regions', 'Подгонка': 'Fitting', 'Результат': 'Results',
    'Фаза': 'Phase', 'Без коррекции': 'No correction', 'abs применяется к': 'Apply abs to',
    'Начальная фаза P0, °': 'Initial P0, °', 'Начальная фаза P1, °': 'Initial P1, °',
    'Окно peak_minima, точки': 'peak_minima window, points', 'Привязка ppm': 'ppm referencing',
    'Из метаданных Bruker': 'Bruker metadata', 'Самый сильный пик': 'Strongest peak',
    'Пик в заданном ppm-окне': 'Peak in ppm window', 'Референс, ppm': 'Reference, ppm',
    'Окно референса: от': 'Reference window: from', 'Окно референса: до': 'Reference window: to',
    'Сдвиг шкалы, ppm': 'ppm offset', 'Удалить цифровой фильтр': 'Remove digital filter',
    'P0/P1 — старт автоматического фазирования. «Без коррекции» показывает реальную часть FFT.\nLB настраивается в отдельном окне.':
        'P0/P1 are initial values for automatic phasing. “No correction” displays the real FFT.\nSet LB in the apodization window.',
    'Threshold (высота пика)': 'Threshold (peak height)',
    'Абсолютная высота после локальной коррекции базовой линии.': 'Absolute height after local baseline correction.',
    'Разделение, точки': 'Min. separation, points', 'Искать и отрицательные пики': 'Include negative peaks',
    'Выключено: A ≥ 0. Включено: амплитуды любого знака + диагностика компенсации.':
        'Off: A ≥ 0. On: signed amplitudes with cancellation diagnostics.',
    'Initial FWHM, Гц': 'Initial FWHM, Hz', 'Минимум FWHM, Гц': 'Min. FWHM, Hz',
    'Максимум FWHM, Гц': 'Max. FWHM, Hz', 'Лимит вычислений / попытку': 'Evaluations per attempt',
    'Попытки с разными ширинами': 'Width initialization attempts', 'Максимум пиков / область': 'Max. peaks per region',
    'Функция ошибки': 'Loss function', 'Предупреждать при RMSE >': 'Warn when RMSE >',
    'Компенсация: порог 0…1': 'Cancellation threshold, 0…1',
    'Центры фиксированы по peak-picking.\nШирина модели = HWHM (ppm); старт = max(LB, initial FWHM) / (2 × SFO1).\nИнтеграл = Σ A × HWHM; полная площадь Лоренца = π × этот интеграл.':
        'Peak-picked centers are fixed.\nModel width = HWHM (ppm); initial width = max(LB, initial FWHM) / (2 × SFO1).\nIntegral = Σ A × HWHM; full Lorentzian area = π × this integral.',
    'Сигнал': 'Signal', 'Сохранить пики и интегралы CSV…': 'Export peaks and integrals CSV…',
    'Сохранить файл подгонки .spfit…': 'Save fitted spectrum .spfit…',
    'После анализа здесь появятся остаток, ограничения ширин, сходимость и обнаруженные пары компенсирующихся линий.':
        'Residuals, width limits, convergence and cancelling component pairs will appear here after analysis.',
    'Во вкладке «Области» выберите режим, затем выделите диапазон мышью на графике.':
        'Choose a mode in Regions, then drag across a plot to select a range.',
    '1. Сохраните области для базовой линии и подгонки.\n2. Внутри них сохраните области интегрирования.':
        '1. Save regions for baseline correction and fitting.\n2. Save integration regions inside them.',
    'Интеграл': 'Integral', 'Имя': 'Name', 'Границы ppm': 'ppm bounds',
    'Обработка для интеграла': 'Parent fitting region', 'Сохранить область': 'Save region',
    'Изменить выбранную': 'Update selected', 'База': 'Baseline',
    'Удалить область обработки': 'Delete fitting region', 'Удалить область интегрирования': 'Delete integration region',
    'Выделение мышью заполняет границы. Область сохраняется кнопкой. Границы можно вводить вручную в любом порядке.\nБазовая линия применяется только в сохранённых областях обработки.':
        'Dragging fills the bounds; click Save region to keep them. Bounds can also be entered in either order.\nBaseline correction is limited to saved fitting regions.',
    'Цвет области…': 'Region color…', 'Цвет интегрирования': 'Integration color',
    'Повторение анализа с сохранёнными параметрами': 'Batch processing with saved settings',
    'Нажмите «Запомнить → серия» во вкладке одного спектра или загрузите JSON.':
        'Click “Use for batch” in Single spectrum or load JSON settings.',
    'Папка серии': 'Dataset folder', 'CSV результата': 'Output CSV', 'Выбрать…': 'Browse…',
    'Искать эксперименты во вложенных папках': 'Search subfolders',
    'Показать найденные спектры': 'List spectra', 'Анализировать серию → CSV': 'Analyze batch → CSV',
    'Остановить после текущего спектра': 'Stop after current spectrum', 'Создать демо-серию…': 'Create demo dataset…',
    'Показать фит выбранного спектра': 'Review selected fit', 'Открыть результаты CSV…': 'Open results CSV…',
    'Открыть файл подгонки .spfit…': 'Open fit file .spfit…', 'Номер спектра': 'Spectrum number',
    'ID спектра': 'Spectrum ID', 'Статус': 'Status',
    'Каждый спектр обрабатывается независимо. Ошибки не останавливают серию; недостоверные интегралы записываются как NaN. CSV: номер спектра и отдельные числовые колонки областей. Для проверки фита выберите строку и нажмите «Показать фит». Рядом сохраняются JSON и папка .fits с файлами .spfit.':
        'Spectra are processed independently. Errors do not stop the batch; invalid integrals are NaN. CSV contains numeric region columns. Select a spectrum and click Review selected fit. JSON files and a .fits folder are saved alongside it.',
    'Откройте папку эксперимента Bruker или демонстрационный спектр.': 'Open a Bruker experiment folder or the demo spectrum.',
    'Параметры или области изменились. Выполните анализ заново.': 'Settings or regions changed. Run the analysis again.',
    'Папка эксперимента с fid и acqus': 'Experiment folder containing fid and acqus',
    'Загрузка FID…': 'Loading FID…', 'Синтетический спектр': 'Synthetic spectrum',
    'Обновляю спектр и локальную базовую линию…': 'Updating spectrum and local baseline…',
    'Выберите области для анализа.': 'Select regions for analysis.',
    'Выберите область в соответствующей таблице.': 'Select a region in the corresponding table.',
    'Удалить область': 'Delete region', 'Сначала загрузите спектр.': 'Load a spectrum first.',
    'Предыдущая подгонка ещё выполняется.': 'The previous fit is still running.',
    'Подгонка областей… Интерфейс доступен; изменение параметров отменит показ старого результата.':
        'Fitting regions… Changing settings will discard the pending result.',
    'Настройки JSON': 'JSON settings', 'Сначала выполните анализ с текущими параметрами.': 'Run the analysis with the current settings first.',
    'Подгонка Spintegrate': 'Spintegrate fit', 'Корневая папка серии': 'Dataset root folder',
    'Папка для синтетической демо-серии': 'Folder for synthetic demo data',
    'Демо-файлы уже существуют. Выберите другую папку.': 'Demo files already exist. Choose another folder.',
    'Создана синтетическая серия из четырёх FID.': 'Created four synthetic FIDs.',
    'Дождитесь загрузки открываемого CSV перед запуском новой серии.': 'Wait for the CSV to finish loading before starting a batch.',
    'Сначала «Запомнить → серия» или загрузить JSON настроек.': 'Click Use for batch or load JSON settings first.',
    'Спектры не найдены.': 'No spectra found.', 'Выберите путь .csv в существующей папке.': 'Choose a .csv path in an existing folder.',
    'Заменить результаты?': 'Replace results?',
    'CSV, JSON или папка .fits уже существуют. Заменить результаты и файлы подгонок с совпадающими номерами?':
        'CSV, JSON or .fits already exists. Replace results and fit files with matching numbers?',
    'Двойной щелчок по строке открывает сохранённую подгонку. Для просмотра после перезапуска используйте «Открыть результаты CSV…».':
        'Double-click a row to review its saved fit. Use Open results CSV to reopen results later.',
    'Дождитесь окончания или остановите текущую серию перед открытием другого CSV.': 'Finish or stop the current batch before opening another CSV.',
    'Открыть CSV вместе с JSON и папкой .fits': 'Open CSV with its JSON files and .fits folder',
    'Результаты Spintegrate': 'Spintegrate results', 'Чтение результатов серии…': 'Loading batch results…',
    'Открыть сохранённую подгонку без повторного расчёта': 'Open saved fit without recalculation',
    'Загрузка сохранённой подгонки…': 'Loading saved fit…', 'Сначала выберите спектр в таблице серии.': 'Select a spectrum in the batch table first.',
    'Файл .spfit не найден и исходный спектр недоступен. Откройте сохранённый .spfit или верните исходный FID для старой серии.':
        'The .spfit file and original spectrum are unavailable. Open a saved .spfit or restore the original FID.',
    'Для этой серии нет сохранённого фита. Повторный расчёт по её сохранённым настройкам…': 'No saved fit is available. Recalculating with the batch recipe…',
    'Открыто окно проверки подгонки. Настройки одиночного анализа не изменены.': 'Fit review opened. Single-spectrum settings are preserved.',
    'Закрыть Spintegrate': 'Close Spintegrate', 'Серия выполняется. Остановить после текущего спектра и закрыть окно?':
        'A batch is running. Stop after the current spectrum and close?',
    'Область обработки:': 'Fitting region:', 'Сохранить картинку PNG…': 'Export figure…',
    'Экспорт картинки…': 'Export figure…', 'Обзор подгонки': 'Fit review',
    'Аподизация и локальная базовая линия': 'Apodization and local baseline',
    'Аподизация (экспоненциальная)': 'Exponential apodization', 'LB = добавочное уширение FWHM, Гц': 'LB = additional FWHM broadening, Hz',
    'Сравнение с LB=0 при тех же прочих настройках.\nГрафик обновляется после короткой паузы.':
        'Compared with LB=0 and otherwise identical settings.\nThe preview updates after a short pause.',
    'Базовая линия выбранной области': 'Selected region baseline', 'Метод': 'Method',
    'Максимум итераций': 'Maximum iterations', 'Допуск tol': 'Tolerance',
    'Полуокно SNIP / mor': 'SNIP / mor half-window', 'Сглаживание SNIP, точки': 'SNIP smoothing, points',
    'Степень modpoly / imodpoly': 'modpoly / imodpoly order',
    'Взять базу из выбранной области': 'Load selected region settings', 'Применить к выбранной области': 'Apply to selected region',
    'Применить ко всем областям': 'Apply to all regions',
    'Изменения здесь задают базу для новых областей. Для сохранённых областей нажмите «Применить». LB применяется сразу.':
        'These baseline settings are defaults for new regions. Click Apply to update existing regions. LB applies immediately.',
    'Выберите сохранённую область в главном окне.': 'Select a saved region in the main window.',
    'Сначала сохраните область обработки.': 'Save a fitting region first.',
    'Аподизация: текущая область': 'Apodization: selected region', 'Аподизация: весь спектр': 'Apodization: full spectrum',
    'Исходная область': 'Original region', 'После коррекции': 'Corrected data',
    'Предпросмотр готов; база применяется только внутри выбранной области.': 'Preview ready; baseline correction is limited to the selected region.',
    'Выберите сохранённую область обработки': 'Select a saved fitting region',
    'Аподизация обновлена. Сохраните область для предпросмотра базовой линии.': 'Apodization updated. Save a region to preview its baseline.',
    'Локальная базовая линия': 'Local baseline', 'Выберите сохранённую область обработки.': 'Select a saved fitting region.',
    'Весь спектр': 'Full spectrum', 'Выбранная область': 'Selected region',
    'Остаток: данные − подгонка': 'Residual: data − fit', 'Химический сдвиг (ppm)': 'Chemical shift (ppm)',
    'Интенсивность (отн. ед.)': 'Intensity (a.u.)', 'После базовой линии': 'Baseline-corrected data',
    'Сумма Лоренцев': 'Lorentzian sum', 'Peak-picking (фикс. центры)': 'Peak-picked positions (fixed)',
    'До базовой линии': 'Before baseline correction', 'Базовая линия': 'Baseline', 'Спектр': 'Spectrum',
    'сошлось': 'converged', 'НЕ СОШЛОСЬ': 'NOT CONVERGED',
    'Экспорт рисунка': 'Figure export', 'Формат': 'Format', 'Качество, DPI': 'Resolution, DPI',
    'Ширина, мм': 'Width, mm', 'Высота, мм': 'Height, mm', 'Язык рисунка': 'Figure language',
    'Как в программе': 'Use interface language', 'Русский': 'Russian', 'Компоновка': 'Layout',
    'Весь спектр + область + остаток': 'Full spectrum + region + residual',
    'Область + остаток': 'Region + residual', 'Только область': 'Region only',
    'Названия областей на рисунке': 'Show region names', 'Заголовок (необязательно)': 'Title (optional)',
    'Сохранить рисунок…': 'Save figure…', 'Отмена': 'Cancel',
    'SVG — векторный формат; DPI влияет на PNG и TIFF.': 'SVG is vector-based; DPI affects PNG and TIFF.',
    'Цвет области должен быть задан как #RRGGBB.': 'Region color must use #RRGGBB format.',
    'Язык должен быть ru или en.': 'Language must be ru or en.',
    'Неверный формат или компоновка рисунка.': 'Invalid figure format or layout.',
    'DPI должен быть целым числом от 72 до 2400.': 'DPI must be an integer from 72 to 2400.',
    'Ширина и высота рисунка должны быть от 40 до 500 мм.': 'Figure width and height must be between 40 and 500 mm.',
    'Рисунок превышает 80 мегапикселей. Уменьшите размер или DPI.': 'Figure exceeds 80 megapixels. Reduce dimensions or DPI.',
    'Пики выше threshold не найдены; интеграл = 0.': 'No peaks exceed the threshold; integral = 0.',
    'Демонстрационные синтетические данные.': 'Synthetic demonstration data.',
    'abs спектра: форма magnitude-пика отличается от чистого Лоренца.': 'Magnitude peak shapes differ from absorption Lorentzians.',
    'abs(FID) до FFT: фазовая/частотная информация FID потеряна.': 'abs(FID) before FFT discards FID phase/frequency information.',
    'Референс = самый сильный пик: проверяйте идентичность сигнала в серии.': 'Reference is the strongest peak; verify its identity across the batch.',
    'Цифровой фильтр отсутствует по метаданным; удаление пропущено.': 'Metadata indicates no digital filter; removal skipped.',
    'Удаление цифрового фильтра отключено.': 'Digital-filter removal is disabled.',
    'Одна или несколько ширин достигли границы ограничений.': 'One or more widths reached their bounds.',
    'Амплитуда компоненты превышает максимум данных более чем в 5 раз.': 'A component amplitude exceeds the data maximum by more than fivefold.',
    'Неизвестный режим фазирования/abs.': 'Unknown phase/abs mode.', 'Неизвестный режим привязки ppm.': 'Unknown ppm reference mode.',
    'Аподизация LB должна быть >= 0 Гц.': 'Apodization LB must be >= 0 Hz.',
    'Окно peak_minima должно быть положительным.': 'peak_minima window must be positive.',
    'Неверное окно референса.': 'Invalid reference window.', 'Threshold должен быть положительным.': 'Threshold must be positive.',
    'Ширины: 0 < min < max; initial > 0.': 'Widths require 0 < min < max and initial > 0.',
    'Лимиты оптимизации и число точек должны быть >= 1.': 'Optimization limits and point counts must be >= 1.',
    'Неверные пороги диагностики.': 'Invalid diagnostic thresholds.',
    'Сохраните области обработки и интегрирования.': 'Save fitting and integration regions first.',
    'Имена областей обработки должны быть уникальны.': 'Fitting-region names must be unique.',
    'Имена интегралов должны быть уникальны.': 'Integral names must be unique.',
    'Настройки должны содержать конечные числа.': 'Settings must contain finite numbers.',
    'Неподдерживаемая версия настроек.': 'Unsupported settings version.',
    'Некорректные параметры базовой линии.': 'Invalid baseline parameters.',
    'Поддерживаются только одномерные Bruker FID.': 'Only one-dimensional Bruker FIDs are supported.',
    'Нужен конечный одномерный FID длиной >= 16 точек.': 'A finite 1D FID with at least 16 points is required.',
    'Неверные SFO1/SW_h/O1 в метаданных.': 'Invalid SFO1/SW_h/O1 metadata.',
    'Окно референса находится вне спектра.': 'Reference window is outside the spectrum.',
    'После обработки появились нечисловые значения.': 'Processing produced nonfinite values.',
    'Область должна содержать >= 8 точек с возрастающей ppm-шкалой.': 'A region needs at least 8 points on an increasing ppm scale.',
    'Коррекция базовой линии вернула нечисловые значения.': 'Baseline correction produced nonfinite values.',
    'Максимальная ширина меньше цифрового разрешения спектра.': 'Maximum width is below the spectral digital resolution.',
    'Выберите существующую папку серии.': 'Choose an existing dataset folder.',
    'Не найдены папки с fid + acqus или демонстрационные .npz.': 'No fid + acqus folders or demo .npz files found.',
    'Неподдерживаемый формат файла подгонки.': 'Unsupported saved-fit format.',
    'Повреждена шкала спектра в файле подгонки.': 'Invalid spectrum axis in saved-fit file.',
    'Неверный список областей в файле подгонки.': 'Invalid region list in saved-fit file.',
    'В CSV отсутствуют столбцы spectrum_number / spectrum_id.': 'CSV is missing spectrum_number / spectrum_id columns.',
    'Подробная диагностика недоступна.': 'Detailed diagnostics are unavailable.',
    'не сохранён': 'not saved', 'Готово': 'Finished', 'Остановлено': 'Stopped',
    'Loss: linear или soft_l1.': 'Loss must be linear or soft_l1.',
    'Zero filling: 1, 2, 4 или 8.': 'Zero filling must be 1, 2, 4 or 8.',
    'peak_minima: недостаточно точек около максимума. Уменьшите окно peak_minima или выберите ACME.':
        'peak_minima: too few points around the maximum. Reduce the window or use ACME.',
    'Автокоррекция фазы достигла лимита оптимизации; проверьте фазу вручную.':
        'Automatic phasing reached its optimization limit; inspect the phase.',
    'Имена spectrum_number и spectrum_id зарезервированы для столбцов CSV.':
        'spectrum_number and spectrum_id are reserved CSV column names.',
    ' №     ppm (фикс.)   A            HWHM ppm     A×HWHM': ' #     ppm (fixed)   A            HWHM ppm     A×HWHM',
    ' №       ppm (фикс.)       A            HWHM ppm        A×HWHM': ' #       ppm (fixed)       A            HWHM ppm        A×HWHM',
}

# Capture already-formatted arguments without changing user-controlled names.
TEMPLATES = {
    'Режим анализа: {message}': 'Analysis method: {message}',
    'Источник спектра: {source}': 'Spectrum source: {source}',
    '{name}: прямое интегрирование, точек {n}': '{name}: direct integration, {n} points',
    'Сохранено областей интегрирования: {n}. Численный интеграл: интенсивность × ppm.':
        'Saved integration regions: {n}. Numerical integral: intensity × ppm.',
    'Сохранены CSV интегралов и JSON настроек: {path}': 'Saved integrals CSV and settings JSON: {path}',
    'Проверка интегрирования — {name}': 'Integration review — {name}',
    'Интеграл {name}: {value} (интенсивность × ppm)': 'Integral {name}: {value} (intensity × ppm)',
    'Не найдены 1r и procs для обработки pdata/{procno}: {path}': 'Missing 1r and procs for pdata/{procno}: {path}',
    'Не найдены 1r/procs для обработки pdata/{procno}.': 'No 1r/procs found for pdata/{procno}.',
    'Проверьте значение поля «{field}».': 'Check the value of “{field}”.',
    'Серия: {n}/{total}, {source} — {status}': 'Batch: {n}/{total}, {source} — {status}',
    'Спектр готов. LB={lb} Гц; P0/P1={phases}°. {message}': 'Spectrum ready. LB={lb} Hz; P0/P1={phases}°. {message}',
    'Выбрано {low}…{high} ppm. Нажмите «Сохранить область».': 'Selected {low}…{high} ppm. Click Save region.',
    'Также удалить {n} связанных областей интегрирования?': 'Also delete {n} linked integration regions?',
    'Статус: {value}': 'Status: {value}', 'Статус спектра: {value}': 'Spectrum status: {value}',
    '{name}: {n} пиков; сошлось={value}': '{name}: {n} peaks; converged={value}',
    'Анализ завершён: {status}. Подробная диагностика во вкладке «Результат».': 'Analysis finished: {status}. See Results for diagnostics.',
    'Настройки и области сохранены: {path}': 'Settings and regions saved: {path}',
    'Параметры запомнены для серии. Последующие изменения одиночного анализа не изменят этот снимок.':
        'Batch recipe saved. Subsequent single-spectrum edits will not change it.',
    'Запомнено: {regions} областей обработки, {integrals} интегралов; фаза={phase}; LB={lb} Гц; threshold={threshold}. Центры и амплитуды определяются заново в каждом спектре.':
        'Recipe: {regions} fitting regions, {integrals} integrals; phase={phase}; LB={lb} Hz; threshold={threshold}. Peaks are detected independently in each spectrum.',
    'Сохранены CSV пиков, CSV интегралов и JSON настроек: {path}': 'Saved peak CSV, integral CSV and settings JSON: {path}',
    'Сохранён файл подгонки: {path}': 'Saved fit: {path}', 'Найдено {n} спектров:': 'Found {n} spectra:',
    '{state}: {n} спектров; CSV: {path}': '{state}: {n} spectra; CSV: {path}',
    'Ошибок: {n}; предупреждений: {m}.': 'Errors: {n}; warnings: {m}.',
    'Файл подгонки: {path}': 'Fit file: {path}',
    'Загружено {n} спектров из {path}. Выберите строку для проверки подгонки.': 'Loaded {n} spectra from {path}. Select a row to review a fit.',
    'Сохранённая подгонка: {path}': 'Saved fit: {path}',
    'ПОВТОРНЫЙ РАСЧЁТ из исходного спектра: {path}': 'RECALCULATED from original spectrum: {path}',
    'Проверка фита Лоренцами — {name}': 'Lorentzian fit review — {name}',
    'Исходный спектр: {path}': 'Original spectrum: {path}',
    'Интеграл {name}: {value} (Σ A × HWHM)': 'Integral {name}: {value} (Σ A × HWHM)',
    'RMSE: {rmse}; сошлось: {value}; nfev: {nfev}': 'RMSE: {rmse}; converged: {value}; nfev: {nfev}',
    'Картинка подгонки сохранена: {path}': 'Figure saved: {path}',
    'Настройки базовой линии сохранены для: {names}': 'Baseline settings saved for: {names}',
    'LB = {lb} Гц': 'LB = {lb} Hz', 'База: {method}': 'Baseline: {method}',
    '{name}: {n} пиков; {state}': '{name}: {n} peaks; {state}',
    '{n} пиков; {state}': '{n} peaks; {state}',
    'Размер: {w} × {h} пикселей': 'Size: {w} × {h} pixels',
    'Размер: {w} × {h} мм (вектор)': 'Size: {w} × {h} mm (vector)',
    'Рисунок сохранён: {path}': 'Figure saved: {path}',
    'Большой остаток: относительный RMSE={value}.': 'Large residual: relative RMSE={value}.',
    'Параметры плохо различимы: condition={value}.': 'Poorly identifiable parameters: condition={value}.',
    'Взаимная компенсация противоположных линий: {pairs}': 'Opposite-sign component cancellation: {pairs}',
    'Суммарная компенсация линий {value}.': 'Total component cancellation: {value}.',
    '{name}: границы области выходят за ppm-диапазон спектра.': '{name}: region bounds are outside the spectrum.',
    '{name}: недостаточно точек для {n} параметров.': '{name}: insufficient data points for {n} parameters.',
    '{name}: область интегрирования должна лежать внутри {parent}.': '{name}: integration region must be inside {parent}.',
    'Не удалось прочитать файл подгонки: {error}': 'Cannot read saved fit: {error}',
    'Не удалось сохранить файл подгонки: {error}': 'Cannot save fit file: {error}',
    '{name}: нужно две различные конечные границы.': '{name}: two different finite bounds are required.',
    '{name}: нужны две различные конечные границы.': '{name}: two different finite bounds are required.',
    '{name}: найдено {n} пиков (лимит {limit}). Повысьте threshold/разделение или сузьте область.':
        '{name}: found {n} peaks (limit {limit}). Increase threshold/separation or narrow the region.',
    '{method}: базовая линия не достигла tol={tol}.': '{method}: baseline did not reach tol={tol}.',
    'Все попытки подгонки завершились ошибкой: {error}': 'All fit attempts failed: {error}',
    'Неизвестный метод базовой линии: {method}': 'Unknown baseline method: {method}',
    'Несогласованные массивы области {name}.': 'Inconsistent arrays for region {name}.',
    'Повреждён массив {name} в файле подгонки.': 'Invalid array {name} in saved-fit file.',
    'Для просмотра CSV нужен сохранённый рецепт рядом с ним: {path}. Можно также открыть отдельный файл .spfit без CSV и JSON.':
        'CSV review requires its recipe alongside it: {path}. A standalone .spfit can be opened without CSV or JSON.',
    'Неверное значение интеграла «{name}» в строке {n}.': 'Invalid integral “{name}” in row {n}.',
}
_FORMATTER = string.Formatter()
_PATTERNS = []
for source, target in TEMPLATES.items():
    parts, fields = [], []
    for literal, field, _, _ in _FORMATTER.parse(source):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append('(.*?)')
            fields.append(field)
    _PATTERNS.append((re.compile('^'+''.join(parts)+'$', re.DOTALL), fields, target))
_REVERSE = {value: key for key, value in EN.items()}


def canonical(text):
    return _REVERSE.get(str(text), str(text))


def tr(text, language='ru'):
    text = str(text)
    if language != 'en':
        return text
    if text in EN:
        return EN[text]
    if '\n' in text:
        return '\n'.join(tr(line, language) for line in text.split('\n'))
    for pattern, fields, target in _PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            values = {k: tr(v, language) if k in {'state', 'message', 'error'} else v
                      for k, v in zip(fields, match.groups())}
            return target.format_map(values)
    if ' | ' in text:
        return ' | '.join(tr(part, language) for part in text.split(' | '))
    for prefix, translated in [('ОШИБКА: ', 'ERROR: '), ('Ошибка: ', 'Error: '),
                               ('ОПТИМИЗАЦИЯ НЕ СОШЛАСЬ: ', 'OPTIMIZATION DID NOT CONVERGE: ')]:
        if text.startswith(prefix):
            return translated + tr(text[len(prefix):], language)
    if ': ' in text:
        prefix, rest = text.split(': ', 1)
        translated = tr(rest, language)
        if translated != rest:
            return prefix + ': ' + translated
    return text


def widget_language(widget):
    while widget is not None:
        if hasattr(widget, '_spintegrate_language'):
            return widget._spintegrate_language
        widget = getattr(widget, 'master', None)
    return 'ru'


class TranslatedVar(tk.StringVar):
    def __init__(self, master, value=''):
        self.owner = master
        self.source = str(value)
        super().__init__(master=master, value=value)
        self.source = str(value)
        self.refresh()

    def set(self, value):
        self.source = str(value)
        super().set(tr(value, widget_language(self.owner)))

    def refresh(self):
        super().set(tr(self.source, widget_language(self.owner)))


def set_label(widget, text):
    widget._source_label = str(text)
    widget.configure(text=tr(text, widget_language(widget)))


def translate_widgets(widget, language):
    """Translate bound UI labels, never data cells, Entry values or region names."""
    if isinstance(widget, (tk.Tk, tk.Toplevel)):
        if not hasattr(widget, '_source_title'):
            widget._source_title = widget.title()
        widget.title(tr(widget._source_title, language))
    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
        if not str(widget.cget('textvariable')):
            if not hasattr(widget, '_source_label'):
                widget._source_label = str(widget.cget('text'))
            widget.configure(text=tr(widget._source_label, language))
    if isinstance(widget, ttk.Notebook):
        for tab in widget.tabs():
            cache = getattr(widget, '_source_tabs', {})
            cache.setdefault(tab, widget.tab(tab, 'text'))
            widget.tab(tab, text=tr(cache[tab], language))
            widget._source_tabs = cache
    if isinstance(widget, ttk.Treeview):
        cache = getattr(widget, '_source_headings', {})
        for column in ('#0', *widget['columns']):
            if str(column).startswith('area'):
                continue  # area column headings are user-provided names
            cache.setdefault(column, widget.heading(column, 'text'))
            widget.heading(column, text=tr(cache[column], language))
        widget._source_headings = cache
    if isinstance(widget, tk.Text) and hasattr(widget, '_source_text'):
        state = str(widget.cget('state'))
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', tr(widget._source_text, language))
        widget.configure(state=state)
    # FieldPanel registers only label choices; parent-region combo values are data.
    if hasattr(widget, 'translate_choices'):
        widget.translate_choices(language)
    for child in widget.winfo_children():
        if isinstance(child, tk.Toplevel) and getattr(child, '_spintegrate_source_window', False):
            continue  # independent source workspaces keep their own language/settings
        translate_widgets(child, language)
