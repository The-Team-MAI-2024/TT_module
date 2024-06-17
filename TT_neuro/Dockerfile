# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем зависимости
RUN pip install numpy tensorflow zmq

# Копируем файлы агента в контейнер
COPY agent.py /app/agent.py

# Переходим в рабочую директорию
WORKDIR /app
# Запускаем агент
CMD ["python", "agent.py"]
