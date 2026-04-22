# OLMoE++

Простой рабочий репозиторий для экспериментов с MoE (Mixture of Experts) и кастомным роутером поверх OLMoE.

## Быстрый старт

```bash
git clone https://github.com/2Leo1/megaOLMoE.git
cd OLMoE++
bash install.sh
```

Запуск окружения:

```bash
./start_env.sh
```

---

## Как это работает

* Всё окружение поднимается в Docker (CUDA, PyTorch и т.д.)
* Локально ничего не засоряется
* У всех в команде одинаковая среда

`install.sh`:

* создает структуру проекта
* клонирует OLMo и OLMoE
* использует наш локальный `megablocks`
* собирает Docker image

Работать нужно **внутри контейнера**.

---

## Структура проекта

```
OLMoE++/
├── install.sh
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── megablocks/     # НАШ КОД
│   ├── OLMo/           # upstream
│   └── OLMoE/          # upstream
├── configs/
├── data/
├── logs/
└── scripts/
```

---

## Где мы реально работаем

Вся работа происходит здесь:

```
src/megablocks/megablocks/layers/
```

Главные файлы:

### router.py

* логика роутинга (куда отправлять токены)
* основное место для экспериментов (L2R, SIPS и т.д.)

### moe.py

* сама MoE-слойка
* берет output роутера и гоняет данные через экспертов

### arguments.py

* конфиг (dataclass)
* все параметры модели и роутера

---

## Тесты

```
src/megablocks/tests/layers/
```

Тут можно:

* смотреть примеры
* проверять свою логику
* писать новые тесты

---

## Как работать

1. Меняешь логику в:

   ```
   router.py
   ```

2. При необходимости правишь:

   ```
   arguments.py
   ```

3. Проверяешь через тесты

4. Если что-то менялось сильно:

   ```bash
   docker compose build
   ```

---

## Важно

* Меняем только `megablocks`
* `OLMo` и `OLMoE` не трогаем
* Всё запускаем через Docker
