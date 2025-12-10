### **Reporte de Workflow e Infraestructura del Proyecto**

El flujo de trabajo se estructura en tres pilares fundamentales: gestión de dependencias con  **uv** , control de versiones de código con  **Git/GitHub** , y control de versiones de datos con  **DVC** .

#### **1. Gestión de Dependencias y Entorno: uv**

**Para garantizar la consistencia entre los entornos de desarrollo de los integrantes, se utiliza ****uv** como gestor de paquetes y proyectos Python^^.

* **Características:** Es una herramienta escrita en Rust, destacada por ser extremadamente rápida (entre 10 y 100 veces más veloz que `<span class="citation-70">pip</span>`)^^.
* **Función en el proyecto:**
  * **Reproducibilidad:** Asegura que todos los miembros del equipo tengan exactamente las mismas versiones de las librerías, eliminando el problema común de "en mi máquina sí funciona"^^.
  * **Gestión de Python:** Administra la versión de Python del proyecto de forma automática, evitando conflictos con la instalación del sistema operativo^^.

#### **2. Control de Versiones y Colaboración: Git y GitHub**

El manejo del código fuente se realiza mediante la integración de un "motor" local y una plataforma en la nube, utilizando una estrategia de ramificación específica.

* **Git (Motor Local):** Se encarga de gestionar el código fuente (`<span class="citation-67">src/</span>`) y las configuraciones locales^^^^^^^^.

  * *Limitación:* Se utiliza exclusivamente para archivos de texto ligero como `<span class="citation-66">.py</span>`, `<span class="citation-66">.md</span>`, `<span class="citation-66">.yaml</span>` y `<span class="citation-66">.ipynb</span>`^^.
* **GitHub (Nube Colaborativa):** Funciona como la plataforma para centralizar el trabajo del equipo^^.
* **Estrategia de Ramificación (Gitflow Simplificado):** Se implementó una estructura de ramas organizada para mantener el orden^^:

  * `<span class="citation-63">main</span>`: Rama de Producción^^.
  * `<span class="citation-62">develop</span>`: Rama de Integración^^.
  * `<span class="citation-61">feat/...</span>`: Ramas individuales por cada integrante para el desarrollo de nuevas características^^.

#### **3. Gestión de Datos Masivos: DVC (Data Version Control)**

**Dado que Git no soporta archivos pesados (como miles de audios **`<span class="citation-60">.wav</span>` o imágenes) y existe el riesgo de desincronización de los datasets limpios entre los miembros, se implementó **DVC**^^.

* **Solución de Almacenamiento Híbrido:**
  * **Git:** Almacena solo archivos "puntero" ligeros (ej. `<span class="citation-59">audio.dvc</span>` de 1KB)^^.
  * **DVC + Google Drive:** Se utiliza la nube (Drive) para guardar los archivos reales que pesan Gigabytes^^.
* **Beneficio Principal:** Permite vincular versiones exactas de código con versiones exactas de los datos, garantizando la reproducibilidad total del experimento^^.
* **Flujo de Comandos:**
  1. `<span class="citation-56">dvc add data/</span>`: Crea el archivo puntero localmente^^.
  2. `<span class="citation-55">dvc push</span>`: Sube los archivos pesados a Google Drive^^.
  3. `<span class="citation-54">git add data.dvc</span>`: Versiona el puntero en Git para compartirlo con el equipo^^.
