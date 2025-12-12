# Reporte de Limpieza y Organización

**Fecha:** 12 de Diciembre, 2025
**Rama Actual:** `develop`

## 1. Organización de Notebooks (`notebooks/`)

Se ha estandarizado la nomenclatura de los notebooks para reflejar un flujo de trabajo secuencial y claro. Los notebooks experimentales o borradores se han movido a una subcarpeta dedicada.

### Cambios Realizados:

| Nombre Anterior | Nuevo Nombre | Descripción |
| :--- | :--- | :--- |
| `0_spotipy.ipynb` | **`00_spotify_api_setup.ipynb`** | Configuración inicial de credenciales y API. |
| `1_EDAlastfm_NuevoDataset.ipynb` | **`01_eda_lastfm_dataset.ipynb`** | EDA y generación del dataset fusionado (`lastfm_spotify_merged.csv`). |
| `item_mapper.ipynb` | **`01b_generate_item_mapper.ipynb`** | Generación del mapeo de IDs (`item_id_mapper.json`). |
| `2_VerificarNuevoDataset.ipynb` | **`02_verify_dataset.ipynb`** | Verificación de integridad del dataset. |
| `3_map_lyrics.ipynb` | **`03_map_lyrics.ipynb`** | Procesamiento y mapeo de letras de canciones. |
| `4_test_divide-dataset.ipynb` | **`04_split_dataset.ipynb`** | División del dataset (train/test/val) o muestreo. |
| `5-verify_user_liking.ipynb` | **`05_verify_user_liking.ipynb`** | Verificación de preferencias de usuario (post-análisis). |
| `Lyrics_.ipynb` | **`experimental/lyrics_download_draft.ipynb`** | Borrador del script de descarga de letras (contiene rutas absolutas y tokens). |

### Estado de Limpieza:
*   Se verificó que los notebooks principales sigan una secuencia lógica.
*   Se creó la carpeta `notebooks/experimental/` para scripts no definitivos.

## 2. Archivos Temporales y `.gitignore`

### Verificación de `data/audio/processed` y `temp`:
*   **Estado:** Estos directorios contienen archivos de audio (`.mp3`) y temporales.
*   **Control de Versiones:**
    *   Están **ignorados por Git** (via `data/` en `.gitignore`).
    *   Están **ignorados por DVC** (via `.dvcignore`).
*   **Conclusión:** La configuración es **correcta**. Estos archivos son artefactos intermedios (audios crudos) que no deben versionarse. El pipeline genera los espectrogramas en `data/audio/mels`, los cuales SÍ están versionados por DVC. El código (`src/data/audio_downloader.py`) se encarga de crear estos directorios si no existen, por lo que no es necesario agregar archivos `.gitkeep`.

### Actualización de `.gitignore` para Reporte:
*   Se actualizó el archivo `.gitignore` para reflejar el cambio de nombre de la carpeta `Reporte/` a `report/`.
*   Se aseguró que los PDFs generados en `report/Out/` sean permitidos (excepción a la regla de ignorar `*.pdf`), mientras que los archivos auxiliares de LaTeX (`.aux`, `.log`) siguen siendo ignorados.

## Próximos Pasos
*   Validar que los scripts en `src/` apunten a las rutas correctas si alguno dependía de los nombres antiguos de los notebooks (aunque los notebooks suelen ser consumidores, no dependencias).
*   Ejecutar el pipeline de prueba para asegurar que la estructura de datos funciona como se espera.
