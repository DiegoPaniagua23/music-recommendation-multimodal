# 🎵 Arquitectura Híbrida Two-Tower para Recomendación Musical Multimodal

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-purple)
![Status](https://img.shields.io/badge/Status-Development-yellow)

---

## 📖 Descripción General

Este proyecto implementa un sistema de recomendación musical del Estado del Arte (SOTA) utilizando una arquitectura **Two-Tower** con fusión **Cross-Modal**. El objetivo es resolver los problemas de escasez de datos (*sparsity*) y brecha semántica (*semantic gap*) en los sistemas tradicionales.

El modelo alinea dos espacios vectoriales:
1.  **User Tower:** Codifica la secuencia histórica de interacciones del usuario usando **SASRec** (Transformer secuencial).
2.  **Item Tower:** Codifica el contenido de la canción mediante **Atención Cruzada (Cross-Attention)** entre Audio (Mel-Spectrograms), Texto (Lyrics) e Imagen (Carátulas).

## 🏗️ Arquitectura del Sistema


## 🚀 Instalación y Configuración

Este proyecto utiliza **`uv`** para la gestión de dependencias y **DVC** para el control de versiones de datos.

### Prerrequisitos

  * Python 3.9+
  * [uv](https://github.com/astral-sh/uv) instalado.
  * `ffmpeg` instalado en el sistema (para procesamiento de audio).

### 1\. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-mir
```

### 2\. Instalar dependencias

```bash
uv sync
# Esto creará el entorno virtual e instalará todo lo necesario
```

### 3\. Configurar Datos (DVC)



### 4\. Variables de Entorno



-----

## 📂 Estructura del Proyecto



-----

## 🤝 Flujo de Trabajo Colaborativo



-----

## 👥 Equipo y Roles



## 📜 Licencia


