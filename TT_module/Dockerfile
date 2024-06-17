# глобальный параметр, общий для сборочного и целевого контейнера
#TODO дистрибутив linux должен быть тем же, что и в целевом образе
ARG ARG_NGINX_VERSION=1.27.0 

# сборочный временный контейнер
FROM alpine:latest AS build-stage

ARG ARG_NGINX_VERSION
ENV NGINX_VERSION=$ARG_NGINX_VERSION

# RUN apk update
# RUN apk add --upgrade apk-tools
# RUN apk upgrade --available
RUN apk add wget git tar

# исходный код nginx
RUN wget https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz
RUN tar -zxvf nginx-${NGINX_VERSION}.tar.gz

# исходный код модуля
COPY ngx_http_upstream_rl_balancer_module/config /usr/share/ngx_http_upstream_rl_balancer_module/config
COPY ngx_http_upstream_rl_balancer_module/ngx_http_upstream_rl_balancer_module.c /usr/share/ngx_http_upstream_rl_balancer_module/ngx_http_upstream_rl_balancer_module.c

# инструменты сборки
RUN apk add gcc g++ make

# зависимости nginx
RUN apk add zlib-dev pcre-dev openssl-dev gd-dev
RUN apk add libzmq zeromq zeromq-dev

# сборка модуля, новые модули добавлять через --add-dynamic-module
# модули собираются в каталог /nginx-${NGINX_VERSION}/objs
RUN cd nginx-${NGINX_VERSION} \
    && ./configure --with-compat --add-dynamic-module=/usr/share/ngx_http_upstream_rl_balancer_module/ --with-ld-opt="-lzmq" \
    && make modules

FROM nginx:1.27.0-alpine-slim

ARG ARG_NGINX_VERSION=1.27.0 
ENV NGINX_VERSION=$ARG_NGINX_VERSION
RUN apk add libzmq zeromq zeromq-dev
# копирование собранного модуля из сборочного образа
COPY --from=build-stage /nginx-${NGINX_VERSION}/objs/ngx_http_upstream_rl_balancer_module.so /etc/nginx/modules

# копирование файлов конфигурации, модифицированных для работы нового модуля
RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/nginx.conf
RUN rm -rf /etc/nginx/conf.d
RUN mkdir /etc/nginx/conf.d
COPY default.conf /etc/nginx/conf.d/default.conf