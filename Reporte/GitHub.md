
# Código Fuente y Estructura del Proyecto

Este repositorio aloja la implementación completa del sistema de recomendación musical multimodal. El desarrollo siguió una estrategia de **Gitflow Simplificado**, organizando las características clave en ramas específicas (`feat/...`) que convergen en la rama de integración.

### Enlace al Repositorio

**[music-recommendation-multimodal](https://github.com/DiegoPaniagua23/music-recommendation-multimodal.git)**

### Organización de Ramas (Branches)

El código está segregado funcionalmente en las siguientes ramas activas:

| Rama (Branch)                           | Descripción Funcional                                                                                                                                                |
| :-------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`main` / `develop`**        | **Producción e Integración:** Ramas estables que contienen la versión final del sistema y la orquestación del flujo completo.                               |
| **`feat/two-tower`**            | **Core de la Arquitectura:** Definición del modelo global Two-Tower, incluyendo la lógica de entrenamiento y la función de pérdida *InfoNCE*.             |
| **`feat/user-tower`**           | **Modelado de Usuario:** Implementación de la torre secuencial basada en Transformers (**SASRec**) y embeddings demográficos.                           |
| **`feat/audio-pipeline`**       | **Procesamiento de Señal:** Scripts para la conversión de audio crudo a Espectrogramas Mel y arquitecturas CNN (ResNet) para extracción de características. |
| **`feat/fine-tuning-mdeberta`** | **NLP Avanzado:** Código para el ajuste fino (Fine-Tuning) de *mDeBERTa* y adaptadores *LoRA* para el procesamiento de letras.                             |
| **`feat/eval-metrics`**         | **Validación:** Implementación de las métricas de evaluación de ranking (NDCG@K, Recall@K) utilizadas en el reporte.                                        |

### Cómo colaborar

Para revisar una característica específica, puedes cambiar de rama tras clonar el repositorio:

```bash
git clone [https://github.com/DiegoPaniagua23/music-recommendation-multimodal.git](https://github.com/DiegoPaniagua23/music-recommendation-multimodal.git)
cd music-recommendation-multimodal

# Ejemplo: Para ver el pipeline de audio
git checkout feat/audio-pipeline
```
