### Векторные представления

В этой директории представлены следующие поддиректории и файлы

- `utils`, `load_model` - вспомогательный код
- `deepfashion`, `fashion200k` - код для предобработки датасетов. В различных скриптах и jupyter тетрадях разработаны решения для загрузки и предобработки ихображений, для формирования изображений отдельных сегментов, генерации файлов с данными по категориям и с данными о тройках для обучения с помощью triplet loss.
- `categories` -данные о категориях сегментов в датасетах.
- `train_for_*` - скрипты для обучения моделей на предсказание категорий или на основе triplet loss. Скрипты могут быть тонко настроены с помощью файлов конфигурации в `configs`
- `validate_retrieval_df2.ipynb` - тетрадь для валидации эмбеддера на задаче retrieval на валидационной части deepfashion2.
