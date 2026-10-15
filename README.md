# AI

Colección de notebooks de **Inteligencia Artificial** en Jupyter/Google Colab: procesamiento de lenguaje natural, análisis de sentimientos, web scraping con clustering de texto y modelos de difusión (DDPM).

## Notebooks

| Notebook | Tema | Librerías principales |
|----------|------|-----------------------|
| [`18_Procesamiento_del_lenguaje_natural.ipynb`](18_Procesamiento_del_lenguaje_natural.ipynb) | PLN: WordCloud, expresiones regulares, NLTK (tokenización, stopwords, n-gramas, frecuencias), spaCy (entidades, `displacy`), TextBlob y Wikipedia | `nltk`, `spacy`, `textblob`, `wordcloud`, `wikipedia` |
| [`8_Analisis_sentimiento_despliegue_word2vec.ipynb`](8_Analisis_sentimiento_despliegue_word2vec.ipynb) | Despliegue de un modelo de análisis de sentimientos con Word2Vec: se carga el modelo, se preparan los datos nuevos y se predice | `nltk`, `pickle`, `pandas` |
| [`Web_Scraping_Clustering_beautifulSoup.ipynb`](Web_Scraping_Clustering_beautifulSoup.ipynb) | Web scraping con BeautifulSoup, limpieza y tokenización, y clustering de texto con TF-IDF | `requests`, `bs4`, `nltk`, `transformers` |
| [`ddpm_cars_128.ipynb`](ddpm_cars_128.ipynb) | Entrenamiento de un modelo de difusión (DDPM) que genera imágenes de carros de 128×128, siguiendo la guía [*Train a diffusion model*](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) de Hugging Face con el dataset [`tanganke/stanford_cars`](https://huggingface.co/datasets/tanganke/stanford_cars) | `diffusers`, `accelerate`, `datasets`, `torch` |
| [`training_2_ddpm_cars_128.ipynb`](training_2_ddpm_cars_128.ipynb) | Segunda fase del DDPM: recarga el modelo, el scheduler y el optimizador del notebook anterior, entrena más épocas y guarda snapshots cada 30 épocas | `diffusers`, `accelerate`, `torch` |

## Cómo ejecutarlos

**Google Colab (recomendado):** los primeros tres notebooks tienen un botón *Open in Colab*. Los notebooks de DDPM necesitan GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU*).

**Local:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib nltk spacy textblob wordcloud wikipedia \
            beautifulsoup4 requests transformers torch torchvision diffusers accelerate datasets
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
jupyter notebook
```

> Algunos notebooks cargan archivos (modelos `.pkl`, datos) desde Google Drive o desde Colab. Para correrlos en local hay que ajustar esas rutas.
