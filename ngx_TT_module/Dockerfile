# Используем базовый образ Nginx
FROM nginx:alpine

# Устанавливаем зависимости для сборки Nginx
RUN apk add --no-cache \
    build-base \
    libzmq \
    zeromq-dev \
    pcre-dev \
    zlib-dev \
    openssl-dev \
    git
COPY zmq.h /usr/include/
# Клонируем исходный код Nginx и ваш модуль балансировки
WORKDIR /usr/share/ngx_http_rl_balancer_module
COPY ngx_http_rl_balancer_module.c ngx_http_rl_balancer_module.c
COPY config config

WORKDIR /usr/src/
RUN wget http://nginx.org/download/nginx-1.18.0.tar.gz
RUN tar -xvf nginx-1.18.0.tar.gz

# Собираем Nginx с вашим модулем балансировки
WORKDIR /usr/src/nginx-1.18.0
RUN ./configure --add-module="/usr/share/ngx_http_rl_balancer_module" --with-cc-opt="-I/usr/include/" --with-ld-opt="-lzmq"
RUN make && make install

# Копируем конфигурационные файлы Nginx
RUN mv /usr/src/nginx-1.18.0/objs/nginx /usr/sbin/nginx 
COPY nginx.conf /etc/nginx/nginx.conf

# Открываем порты
EXPOSE 80

# Запускаем Nginx
ENTRYPOINT ["nginx", "-g", "daemon off;"]
