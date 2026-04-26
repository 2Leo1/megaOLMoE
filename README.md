# megaOLMoE

Форк [megablocks](https://github.com/Muennighoff/megablocks/tree/olmoe) для экспериментов с роутером OLMoE. Цель — воспроизвести, а затем улучшить [L2R](https://arxiv.org/pdf/2601.21349) вместо стандартного top-k роутинга.

## Установка

```bash
git clone https://github.com/2Leo1/megaOLMoE.git
cd megaOLMoE
bash install.sh      # клонирует OLMo/OLMoE, собирает Docker-образ
./start_env.sh       # войти в контейнер
```

## Структура

```
src/megablocks/megablocks/layers/
├── moe.py
├── routerL2R.py
├── routerBasick.py
└── arguments.py
```

## Текущая структура роутера

- Проекция токенов в низкоранговое пространство (`hidden_size → latent_size`)
- Несколько анкер-векторов на эксперта
- SIPS-логиты: скор = амплитуда(q) × амплитуда(k) × cos(q, k), агрегация через logsumexp по анкерам
- RMSNorm на входе роутера, jitter при обучении
