#!/bin/bash

# Crear estructura de directorios
echo "Creando carpetas..."
mkdir -p Chapters
mkdir -p Figures/Tikz
mkdir -p Out

# Mover capítulos a la carpeta Chapters
echo "Moviendo capítulos..."
mv introduccion.tex Chapters/ 2>/dev/null
mv planeacion.tex Chapters/ 2>/dev/null
mv triz.tex Chapters/ 2>/dev/null
mv relacionados.tex Chapters/ 2>/dev/null
mv dataset.tex Chapters/ 2>/dev/null
mv modelo.tex Chapters/ 2>/dev/null
mv resultados.tex Chapters/ 2>/dev/null
mv discusion.tex Chapters/ 2>/dev/null
mv conclusiones.tex Chapters/ 2>/dev/null

# Mover archivos TikZ a Figures/Tikz
echo "Moviendo figuras TikZ..."
mv item_tower.tex Figures/Tikz/ 2>/dev/null
mv two-towers.tex Figures/Tikz/ 2>/dev/null
mv user_tower.tex Figures/Tikz/ 2>/dev/null

# Mover el PDF compilado si existe
if [ -f "main.pdf" ]; then
    echo "Moviendo main.pdf a Out/..."
    mv main.pdf Out/
fi

echo "Organización completada."
