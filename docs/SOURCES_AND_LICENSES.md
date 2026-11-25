# Источники и сторонние компоненты

Сведения проверены при подготовке версии 0.1.0 24 ноября 2026 года. Новые
обновления библиотек не подставляются автоматически; requirements фиксирует
основные версии. Фактически проверенное CPU-окружение Linux/Python 3.12
дополнительно записано в requirements-verified-linux-cpu.txt. Этот файл не
является универсальным lock для Windows/CUDA.

## Базовая модель

| Объект | Источник |
|---|---|
| Исходная модель | https://huggingface.co/apple/MobileCLIP2-S0 |
| Официальный код и статья | https://github.com/apple-aiml-research/ml-mobileclip |
| Каноническая адаптация timm | https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP |
| Ревизия адаптации | `095906d28bf54d7584dc411e8ffe448f34289e05` |
| Файл источника | `open_clip_model.safetensors` |
| SHA-256 источника | `ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95` |
| Источник лицензии | https://huggingface.co/apple/MobileCLIP2-S0/blob/3136ea51c8ed56b9f9abfab04cb816735aaad6cb/LICENSE |

В models/base включены полный текст лицензии, attribution, исходная OpenCLIP
конфигурация и base.json с происхождением/хешами. Изменение относительно полного
checkpoint: сохранены только `visual.trunk` тензоры с удалением префикса.
Текстовый энкодер удалён. Дообучение базы на телефонах не выполнялось.

Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement.

Лицензия Apple ограничена некоммерческими научными исследованиями и
академической разработкой. Передача производных весов сопровождается её
текстом, attribution и описанием изменений. Для коммерческого применения
этот стартовый выбор базы использовать не следует; потребуется иная основа.
Адаптация timm не отменяет исходную лицензию весов Apple.

## Основные библиотеки

Библиотеки устанавливаются отдельно, не упакованы в ZIP как готовое окружение.
Их лицензии не заменяются лицензией базы.

| Компонент | Версия | Лицензия проекта и источник |
|---|---|---|
| Python | Рекомендуется 3.12 x64 | PSF, https://www.python.org/psf/license/ |
| PyTorch | 2.8.0 | BSD-3-Clause, https://github.com/pytorch/pytorch/blob/main/LICENSE |
| torchvision | 0.23.0 | BSD-3-Clause, https://github.com/pytorch/vision/blob/main/LICENSE |
| timm | 1.0.20 | Apache-2.0, https://github.com/huggingface/pytorch-image-models |
| safetensors | 0.6.2 | Apache-2.0, https://github.com/huggingface/safetensors |
| NumPy | 2.2.6 | BSD-3-Clause, https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| Pillow | 12.0.0 | HPND и включённые уведомления, https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| scikit-learn | 1.7.2 | BSD-3-Clause, https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| SciPy | 1.16.3 | BSD-3-Clause, https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| huggingface-hub | 0.35.3 | Apache-2.0, https://github.com/huggingface/huggingface_hub |
| psutil | 7.1.2 | BSD-3-Clause, https://github.com/giampaolo/psutil |

OpenCLIP 3.2.0 (MIT, https://github.com/mlfoundations/open_clip) использован
только для независимого сравнения выходов при проверке поставки; в рабочих
зависимостях проекта его нет. Вторичные зависимости перечислены в lock-снимке.
Команды установки выбранных сборок PyTorch:
https://pytorch.org/get-started/previous-versions/ .

## Исходные вложения

В references сохранены неизменённые исходные ТЗ и учебный архив. Их материалы
используются только как предоставленные пользователем исходные примеры;
новые разрешения на публикацию или переиспользование чужих фотографий не
предполагаются. Для собираемой выборки участники самостоятельно фиксируют
источники и допустимость использования в столбце source.
