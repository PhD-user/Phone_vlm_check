# Локальное распознавание iPhone и Android

Версия проекта 0.1.0. Это стартовый комплект для продолжения разработки и
обучения другими участниками. Реализован полный цикл: разметка → проверка →
групповое разбиение → извлечение признаков → обучение → тест → распознавание.

**Начните с этого файла, затем прочитайте `docs/TRAINING.md`.**

В `models/base` включён предобученный визуальный энкодер MobileCLIP2-S0. Он
не обучался на ваших фотографиях телефонов. Фотографий смартфонов и готовой
головы iPhone / Android в комплекте нет. До обучения команды `predict` и `gui`
не смогут загрузить `runs/experiment_001/run.json` — это ожидаемо.

## Принятые решения

- Основа — MobileCLIP2-S0, визуальный энкодер FastViT MCi0, вход 256×256,
  выход 512 признаков. В поставке оставлена только визуальная часть модели.
- Обучение первой версии — логистическая регрессия на L2-нормированных
  признаках замороженной нейросети. Полное дообучение энкодера и LoRA не реализованы.
- Рабочие классы: `android` и `iphone`, именно в таком порядке.
- Назначение — распознавание внешнего типа телефона. Установленную ОС по
  фотографии корпуса достоверно определить нельзя.
- Обучение признаков можно выполнить на CPU или NVIDIA. Ответ о доступном
  оборудовании принят как NVIDIA с 6–8 ГБ VRAM; конкретная модель пока неизвестна.
- Применение по умолчанию на CPU. Рабочий режим не делает сетевых запросов.
- Лицензия базовых весов — Apple Machine Learning Research Model License,
  только исследовательские и академические задачи. Полный текст включён в
  `models/base/APPLE_MODEL_LICENSE.txt`.

## Требования и установка

Рекомендуется **Python 3.12 x64**, Windows 10/11 или Linux x86-64, 8 ГБ ОЗУ.
Проверка выполнена на Linux с Python 3.12 и CPU; Windows, GUI и CUDA нужно
проверить на компьютере участников. Зависимости устанавливаются через интернет.
Предусмотрите 4–8 ГБ места для окружения, особенно при установке CUDA-сборки;
это требование к среде разработки, а не подтверждение размера готовой поставки.

Распакуйте архив полностью. Команды выполняйте **из папки `phone_vlm_starter`**.
Не запускайте код из окна просмотра ZIP.

### Windows

Установите Python 3.12 с поддержкой Tkinter и launcher `py`. Затем откройте
`INSTALL_CPU.bat`. Для NVIDIA можно запустить из PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Device cuda
```

Ручная установка CPU в PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m phone_vlm doctor
```

Для NVIDIA в команде установки PyTorch замените `/cpu` на `/cu126`.
Если меняете уже установленную сборку, используйте точные версии
`torch==2.8.0+cu126` и `torchvision==0.23.0+cu126`; для обратного перехода —
суффикс `+cpu`. Скрипты установки делают этот выбор автоматически.
Отдельная установка CUDA Toolkit для готовых колёс PyTorch обычно не нужна;
совместимый драйвер NVIDIA необходим. Если doctor показывает
`cuda_available: false`, используйте CPU или проверьте драйвер/сборку PyTorch.

### Linux

```bash
bash scripts/setup_linux.sh cpu
# Для NVIDIA: bash scripts/setup_linux.sh cuda
```

Ручной вариант:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m phone_vlm doctor
```

В следующих примерах `python` обозначает **интерпретатор созданного окружения**.
Для активации: PowerShell `.venv\Scripts\Activate.ps1`, CMD
`.venv\Scripts\activate.bat`, Linux `source .venv/bin/activate`.
Если активация запрещена настройками PowerShell, используйте полный путь
`.venv\Scripts\python.exe` вместо `python`; менять настройки не требуется.

## Порядок первого обучения

1. Положите фотографии в `data/raw/android` и `data/raw/iphone`.
2. Создайте черновик CSV и вручную заполните идентификаторы устройств/сессий.
3. Проверьте файлы и похожие пары, затем создайте разбиение.
4. Извлеките признаки, обучите голову, отдельно проведите итоговый тест.

```bash
python -m phone_vlm index
# ЗАПОЛНИТЕ device_id и session_id в data/labels.csv перед следующей командой.
python -m phone_vlm audit
# ПРОВЕРЬТЕ data/audited.audit.json. При дублях исправьте group_id в labels.csv
# и повторите audit с новым выходным файлом: подробности в docs/TRAINING.md.
python -m phone_vlm split
python -m phone_vlm extract --device auto
python -m phone_vlm train --run runs/experiment_001
python -m phone_vlm evaluate --run runs/experiment_001 --output reports/test_001
```

Сначала прочитайте полную инструкцию: комментарии в этом блоке — обязательные
ручные шаги, их нельзя просто пропустить при копировании команд.

Если базовые веса удалены, восстановите их командой `python -m phone_vlm download`.
Это отдельная явная загрузка из сети. При наличии исправной базы команда
использует её повторно. Полный исходный checkpoint временно требует около 300 МБ.

## Применение после обучения

```bash
python -m phone_vlm predict data/raw/iphone/example.jpg --run runs/experiment_001
python -m phone_vlm batch inputs --output reports/batch_001.csv
python -m phone_vlm gui --run runs/experiment_001
python -m phone_vlm benchmark --run runs/experiment_001 --device cpu --output reports/benchmark_001
```

На Windows после обучения можно открыть `START_GUI.bat`. Папку `inputs` для
пакетной проверки создайте самостоятельно или передайте существующую папку.
Некорректный файл даёт ошибку своей строки, остальные файлы продолжают обрабатываться.

Порог предупреждения по умолчанию 0,60. Softmax **не калиброван** и не является
гарантированной вероятностью правильного ответа. Система не обнаруживает
посторонние предметы надёжно: двухклассовая модель может уверенно выбрать один
класс даже для фотографии без телефона. См. `docs/LIMITATIONS.md`.

## Проверка и продолжение работы

```bash
python -m unittest discover -s tests -v
python scripts/verify_archive.py
```

Вторую команду применяйте сразу после распаковки исходного архива: после
собственных правок несовпадение хешей ожидаемо. Состояние проверок этой поставки
зафиксировано в `docs/VALIDATION.md` и `validation/verification.json`.

| Где искать | Содержание |
|---|---|
| `docs/TRAINING.md` | Полная инструкция участникам, данные, команды, смена эксперимента |
| `docs/ARCHITECTURE.md` | Модули, контракты, причины решений, формат модели |
| `docs/USER_GUIDE.md` | Распознавание, статусы, журнал, типичные ошибки |
| `docs/CONTINUE_HERE.md` | Состояние проекта и следующий этап |
| `docs/TZ_CHANGES.md` | Принятые уточнения примерного ТЗ |
| `docs/ACCEPTANCE.md` | Порядок итоговых испытаний и шаблон протокола |
| `docs/SOURCES_AND_LICENSES.md` | Источники весов, ревизии, лицензии |
| `phone_vlm/` | Исходники с русскими комментариями и docstring |
| `references/` | Неизменённые исходные вложения пользователя |

Не пересылайте `.venv`: окружение создаётся заново по инструкции. Сохраняйте
фотографии, CSV разметки/разбиения, config и папку эксперимента. Готовый архив
своего состояния можно собрать через `scripts/make_checkpoint.py`.
