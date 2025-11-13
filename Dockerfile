# Usa TensorFlow preinstalado (rápido y compatible)
FROM tensorflow/tensorflow:2.20.0

# Evita el buffering para ver logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Crea y usa el directorio de trabajo
WORKDIR /app

# Copia dependencias primero (para aprovechar la cache)
COPY requirements.txt .

# Instala dependencias
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Render asigna automáticamente un puerto mediante la variable PORT
EXPOSE 10000

# Comando para ejecutar Flask usando el puerto dinámico
CMD ["python", "app.py"]
