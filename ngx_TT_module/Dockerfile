FROM debian:bullseye-slim AS builder

ENV NGINX_VERSION=1.27.0

# Устанавливаем зависимости для сборки Nginx
RUN apt update && \
    apt install -y \
    build-essential \
    curl \
    git \
    libpcre++-dev \
    zlib1g-dev \ 
    libzmq3-dev

RUN curl http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz -o /tmp/nginx-${NGINX_VERSION}.tar.gz && \
    cd /tmp && \
    tar xvzf nginx-${NGINX_VERSION}.tar.gz

# Клонируем исходный код Nginx и ваш модуль балансировки
WORKDIR /usr/share/ngx_http_rl_balancer_module
COPY ngx_http_rl_balancer_module.c ngx_http_rl_balancer_module.c
COPY config config

WORKDIR /
RUN cd /tmp/nginx-${NGINX_VERSION} && \
    ./configure --with-compat --add-module="/usr/share/ngx_http_rl_balancer_module" --with-cc-opt="-I/usr/include/" --with-ld-opt="-lzmq" && \
    make modules

FROM nginx:1.27.0

COPY --from=builder /tmp/nginx-${NGINX_VERSION}/objs/* /etc/nginx/modules/

COPY --from=builder /usr/local/lib/* /lib/x86_64-linux-gnu/
























# # Используем базовый образ Nginx
# FROM nginx:alpine

# # Устанавливаем зависимости для сборки Nginx
# RUN apk add --no-cache \
#     build-base \
#     libzmq \
#     zeromq-dev \
#     pcre-dev \
#     zlib-dev \
#     openssl-dev \
#     git

# # Клонируем исходный код Nginx и ваш модуль балансировки
# WORKDIR /usr/share/ngx_http_rl_balancer_module
# COPY ngx_http_rl_balancer_module.c ngx_http_rl_balancer_module.c
# COPY config config

# WORKDIR /usr/src/
# RUN wget http://nginx.org/download/nginx-1.18.0.tar.gz
# RUN tar -xvf nginx-1.18.0.tar.gz

# # Собираем Nginx с вашим модулем балансировки
# WORKDIR /usr/src/nginx-1.18.0
# RUN ./configure --add-module="/usr/share/ngx_http_rl_balancer_module" --with-cc-opt="-I/usr/include/" --with-ld-opt="-lzmq"
# RUN make && make install

# # Копируем конфигурационные файлы Nginx
# RUN mv /usr/src/nginx-1.18.0/objs/nginx /usr/sbin/nginx 

# # Открываем порты
# EXPOSE 80
# EXPOSE 8080

# # Запускаем Nginx
# CMD ["nginx", "-g", "daemon off;"]
