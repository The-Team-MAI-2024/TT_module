/*
 * Copyright (C) 2024 The Team
 */

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <zmq.h>  // ZeroMQ для IPC

static ngx_int_t ngx_http_upstream_rl_balancer_init(ngx_conf_t *cf, ngx_http_upstream_srv_conf_t *us);
static ngx_int_t ngx_http_upstream_rl_balancer_init_peer(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *us);
static ngx_int_t ngx_http_upstream_rl_balancer_get_peer(ngx_peer_connection_t *pc, void *data);
static char *ngx_http_upstream_rl_balancer(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);

static ngx_command_t ngx_http_upstream_rl_balancer_commands[] = {
    { ngx_string("rl_balancer"),
      NGX_HTTP_UPS_CONF|NGX_CONF_NOARGS,
      ngx_http_upstream_rl_balancer,
      0,
      0,
      NULL },

      ngx_null_command
};

static ngx_http_module_t ngx_http_upstream_rl_balancer_module_ctx = {
    NULL,                                  /* preconfiguration */
    NULL,                                  /* postconfiguration */

    NULL,                                  /* create main configuration */
    NULL,                                  /* init main configuration */

    NULL,                                  /* create server configuration */
    NULL,                                  /* merge server configuration */

    NULL,                                  /* create location configuration */
    NULL                                   /* merge location configuration */
};

ngx_module_t ngx_http_upstream_rl_balancer_module = {
    NGX_MODULE_V1,
    &ngx_http_upstream_rl_balancer_module_ctx,      /* module context */
    ngx_http_upstream_rl_balancer_commands,         /* module directives */
    NGX_HTTP_MODULE,                       /* module type */
    NULL,                                  /* init master */
    NULL,                                  /* init module */
    NULL,                                  /* init process */
    NULL,                                  /* init thread */
    NULL,                                  /* exit thread */
    NULL,                                  /* exit process */
    NULL,                                  /* exit master */
    NGX_MODULE_V1_PADDING
};

typedef struct {
    ngx_http_upstream_rr_peer_data_t rrp;
} ngx_http_upstream_rl_balancer_peer_data_t;

static ngx_int_t
ngx_http_upstream_rl_balancer_init(ngx_conf_t *cf, ngx_http_upstream_srv_conf_t *us)
{
    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, cf->log, 0, "init rl balancer");

    if (ngx_http_upstream_init_round_robin(cf, us) != NGX_OK) {
        return NGX_ERROR;
    }

    us->peer.init = ngx_http_upstream_rl_balancer_init_peer;

    return NGX_OK;
}

static ngx_int_t
ngx_http_upstream_rl_balancer_init_peer(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *us)
{
    ngx_http_upstream_rl_balancer_peer_data_t *rlp;

    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "init rl balancer peer");

    rlp = ngx_palloc(r->pool, sizeof(ngx_http_upstream_rl_balancer_peer_data_t));
    if (rlp == NULL) {
        return NGX_ERROR;
    }

    r->upstream->peer.data = &rlp->rrp;

    if (ngx_http_upstream_init_round_robin_peer(r, us) != NGX_OK) {
        return NGX_ERROR;
    }

    r->upstream->peer.get = ngx_http_upstream_rl_balancer_get_peer;

    return NGX_OK;
}

static ngx_int_t
ngx_http_upstream_rl_balancer_get_peer(ngx_peer_connection_t *pc, void *data)
{
    ngx_http_upstream_rl_balancer_peer_data_t *rlp = data;
    ngx_http_upstream_rr_peer_t *peer;
    ngx_http_upstream_rr_peers_t *peers = rlp->rrp.peers;

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, pc->log, 0, "get rl balancer peer, try: %ui", pc->tries);

    if (peers->single) {
        return ngx_http_upstream_get_round_robin_peer(pc, &rlp->rrp);
    }

    pc->cached = 0;
    pc->connection = NULL;

    // Инициализация ZeroMQ
    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);
    zmq_connect(requester, "tcp://localhost:5555");

    // Формируем запрос для агента RL
    char request[256];
    ngx_snprintf((u_char *)request, sizeof(request), "{\"state\": [%d, %d, %d, %d], \"next_state\": [%d, %d, %d, %d], \"reward\": %d, \"done\": %d}",
                 1, 0, 0, 0,  // пример текущего состояния
                 0, 1, 0, 0,  // пример следующего состояния
                 1,           // пример награды
                 0);          // пример завершения эпизода

    zmq_send(requester, request, ngx_strlen(request), 0);

    // Получаем ответ от агента
    char buffer[10];
    zmq_recv(requester, buffer, 10, 0);

    ngx_uint_t peer_index = ngx_atoi((u_char *)buffer, ngx_strlen(buffer));
    zmq_close(requester);
    zmq_ctx_destroy(context);

    // Выбираем сервер по индексу, полученному от агента RL
    ngx_uint_t i;
    for (peer = peers->peer, i = 0; peer; peer = peer->next, i++) {
        if (i == peer_index) {
            pc->sockaddr = peer->sockaddr;
            pc->socklen = peer->socklen;
            pc->name = &peer->name;

            peer->conns++;
            rlp->rrp.current = peer;

            ngx_http_upstream_rr_peer_lock(peers, peer);
            ngx_http_upstream_rr_peer_unlock(peers, peer);

            return NGX_OK;
        }
    }

    ngx_http_upstream_rr_peers_unlock(peers);

    return NGX_BUSY;
}

static char *
ngx_http_upstream_rl_balancer(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_upstream_srv_conf_t *uscf;

    uscf = ngx_http_conf_get_module_srv_conf(cf, ngx_http_upstream_module);

    if (uscf->peer.init_upstream) {
        ngx_conf_log_error(NGX_LOG_WARN, cf, 0, "load balancing method redefined");
    }

    uscf->peer.init_upstream = ngx_http_upstream_rl_balancer_init;

    uscf->flags = NGX_HTTP_UPSTREAM_CREATE
                  |NGX_HTTP_UPSTREAM_WEIGHT
                  |NGX_HTTP_UPSTREAM_MAX_CONNS
                  |NGX_HTTP_UPSTREAM_MAX_FAILS
                  |NGX_HTTP_UPSTREAM_FAIL_TIMEOUT
                  |NGX_HTTP_UPSTREAM_DOWN
                  |NGX_HTTP_UPSTREAM_BACKUP;

    return NGX_CONF_OK;
}

