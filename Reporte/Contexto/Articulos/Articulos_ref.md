### 1. El estándar de la industria (Arquitectura Two-Tower)

Estos artículos son obligatorios para justificar por qué elegiste una arquitectura de dos torres. Son la base de los sistemas de recuperación (retrieval) en **YouTube** y  **Google** .

* **"Deep Neural Networks for YouTube Recommendations" (Covington et al., 2016)**
  * **Contexto:** Este es el *paper* fundacional que introdujo la idea de separar el problema en "Candidat Generation" (Retrieval) y "Ranking". Aunque usaron una arquitectura más simple, establece el flujo de trabajo que tú sigues.
  * **Por qué citarlo:** Para justificar la estructura general de tu sistema y la necesidad de lidiar con millones de ítems (escalabilidad).
* **"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Retrieval" (Yi et al., Google, 2019)**
  * **Contexto:** Este es el artículo que formaliza la arquitectura **Two-Tower** moderna que estás usando. Introduce cómo entrenar eficazmente dos codificadores separados (usuario e ítem) y hacer el *dot product* para la similaridad.
  * **Por qué citarlo:** Es la referencia técnica directa de tu arquitectura global.

### 2. Modelos de Spotify y Audio (Validación de tu Item Tower)

Tu proyecto procesa audio crudo -> Espectrogramas Mel -> CNN (ResNet). **Spotify** fue pionero en esto.

* **"Deep content-based music recommendation" (Van den Oord, Dieleman, Schrauwen, 2013)**
  * **Contexto:** Sander Dieleman (quien luego trabajó en Spotify y DeepMind) demostró que usar CNNs sobre espectrogramas Mel podía predecir factores latentes para recomendación. Es la base científica de tu módulo de audio.
  * **Por qué citarlo:** Valida tu decisión de usar Mel-Spectrograms + CNNs (ResNet) en lugar de usar metadatos manuales. Es el argumento central para solucionar el *Cold-Start* en música.
* **"Recommending Long-tail Music" (Spotify Research)**
  * Aunque a veces no publican *papers* académicos tradicionales, puedes citar trabajos como **"The Long Tail of Recommender Systems"** o referencias a su sistema "BaRT" (Bandits for Recommendations as Treatments) si quisieras hablar de exploración, pero para tu tesis, el de Dieleman (2013) es el más crítico.

### 3. Modelos Secuenciales (Validación de tu User Tower)

Estás usando **SASRec** en tu torre de usuario. Debes citar el origen de esto y su evolución (estilo BERT), que es lo que usan plataformas como **Alibaba** o **TikTok** (indirectamente, a través de modelos de atención secuencial).

* **"Self-Attentive Sequential Recommendation" (Kang & McAuley, 2018)**
  * **Contexto:** El paper original de  **SASRec** . Demuestra que usar mecanismos de auto-atención (Transformers) supera a las RNNs/LSTMs para modelar el historial del usuario.
  * **Por qué citarlo:** Es la cita obligatoria para tu  *User Tower* .
* **"BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer" (Sun et al., 2019)**
  * **Contexto:** La evolución de SASRec usando BERT (bidireccional).
  * **Por qué citarlo:** Sirve para dar contexto en el estado del arte ("State of the Art" o SOTA) en la introducción, mostrando que los Transformers dominan el campo.

### 4. TikTok y Redes Neuronales Profundas (Interacción compleja)

Para mencionar modelos tipo **TikTok** (ByteDance) o sistemas que modelan intereses evolutivos muy rápidos.

* **"Deep Interest Network for Click-Through Rate Prediction" (Zhou et al., Alibaba, 2018)**
  * **Contexto:** Aunque es de Alibaba, este modelo (DIN) y su sucesor (DIEN) introdujeron la idea de atender selectivamente a partes del historial del usuario relevantes para el ítem candidato actual. Es muy similar a la lógica de "scroll infinito" y recomendación inmediata de TikTok.
  * **Por qué citarlo:** Para contrastar tu enfoque. Tu usas *embeddings* fijos por sesión (Two-Tower), mientras que estos modelos hacen atención "target-user". Sirve para enriquecer la sección de "Trabajos Relacionados".
* **"Monolith: Real Time Recommendation System With Collisionless Embedding Table" (Liu et al., ByteDance, 2022)**
  * **Contexto:** Un *paper* técnico de ByteDance (dueños de TikTok) sobre cómo manejan *embeddings* en tiempo real y colisiones de datos.
  * **Por qué citarlo:** Si quieres mencionar explícitamente tecnología de **TikTok/ByteDance** en cuanto a ingeniería de *embeddings* y manejo de  *sparsity* .

### 5. Netflix (Autoencoders y Filtrado Colaborativo)

* **"Variational Autoencoders for Collaborative Filtering" (Liang et al., Netflix/MIT, 2018)**
  * **Contexto:** Introduce  **Mult-VAE** . Netflix usa una variedad de modelos, pero este paper fue muy influyente al mostrar que los Autoencoders Variacionales eran superiores para el filtrado colaborativo implícito.
  * **Por qué citarlo:** Como un ejemplo de arquitecturas profundas alternativas (no secuenciales) que se usan en la industria para "Matrix Factorization" no lineal.

### Resumen de cómo integrarlos en tu Introducción:

Puedes estructurar un párrafo así (ejemplo):

> *"En la industria actual, la arquitectura predominante para la recuperación eficiente de ítems en catálogos masivos es el enfoque  **Two-Tower** , popularizado por **YouTube** [Covington et al., 2016] y perfeccionado por **Google** [Yi et al., 2019]. Para el modelado de preferencias de usuario, los enfoques secuenciales basados en auto-atención, como **SASRec** [Kang & McAuley, 2018], han demostrado superar a los métodos tradicionales, capturando la evolución dinámica de intereses similar a lo observado en plataformas como **TikTok** o **Alibaba** [Zhou et al., 2018]. Sin embargo, en el dominio musical, el problema de arranque en frío ('Cold-Start') persiste. Inspirados por los trabajos pioneros de **Spotify** en el uso de redes convolucionales sobre espectrogramas de audio [Van den Oord et al., 2013], nuestra propuesta integra..."*



### 1. El estándar Two-Tower (YouTube/Google)

Estos son los documentos fundamentales para tu arquitectura global.

* **"Deep Neural Networks for YouTube Recommendations"** (Covington et al., 2016)
  * 📄 **PDF:** [Google Research PDF](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
  * Este es el clásico que define la separación entre *Candidate Generation* y  *Ranking* .
* **"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Retrieval"** (Yi et al., 2019)
  * 📄 **PDF:** [Google Research PDF](https://research.google/pubs/pub48840/) (Busca el botón "Download PDF" en la página) o vía [ACM Digital Library](https://dl.acm.org/doi/10.1145/3298689.3346996).
  * Este formaliza la corrección de sesgo en el *softmax* que es crucial para entrenar Two-Towers correctamente.

### 2. Audio y Música (Spotify)

La base para tu "Item Tower" de procesamiento de audio.

* **"Deep content-based music recommendation"** (Van den Oord, Dieleman, Schrauwen, 2013)
  * 📄 **PDF:** [NIPS Proceedings](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf)
  * El paper original que demostró el uso de CNNs sobre espectrogramas Mel para recomendación.

### 3. Modelos Secuenciales (User Tower)

Las referencias para tu implementación de SASRec.

* **"Self-Attentive Sequential Recommendation"** (SASRec - Kang & McAuley, 2018)
  * 📄 **Link:** [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
  * El paper que introdujo el uso de Transformers para secuencias de usuario.
* **"BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"** (Sun et al., 2019)
  * 📄 **Link:** [arXiv:1904.06690](https://arxiv.org/abs/1904.06690)
  * La evolución bidireccional de SASRec.

### 4. Interacción Compleja y Tiempo Real (Alibaba/TikTok/ByteDance)

Referencias adicionales para la discusión del estado del arte.

* **"Deep Interest Network for Click-Through Rate Prediction"** (DIN - Zhou et al., Alibaba, 2018)
  * 📄 **Link:** [arXiv:1706.06978](https://arxiv.org/abs/1706.06978)
  * Sobre cómo atender a partes específicas del historial del usuario (Atención local vs. Global).
* **"Monolith: Real Time Recommendation System With Collisionless Embedding Table"** (Liu et al., ByteDance, 2022)
  * 📄 **Link:** [arXiv:2209.07663](https://arxiv.org/abs/2209.07663)
  * La arquitectura detrás de TikTok para manejo de embeddings en tiempo real.

### 5. Filtrado Colaborativo Profundo (Netflix)

Alternativa no secuencial.

* **"Variational Autoencoders for Collaborative Filtering"** (Mult-VAE - Liang et al., 2018)
  * 📄 **Link:** [arXiv:1802.05814](https://arxiv.org/abs/1802.05814)
  * El uso de Autoencoders Variacionales para recomendación implícita.
