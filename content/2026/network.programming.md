---
title: network programming: echo server
date: 2026-05-14 00:31
tags: network
---

[TOC]

# 0. intro

# 1. 单线程 echo client-server

思考再三，不知保留全部还是只 skeleton，暂且先这样吧，且听下回分解。

## 1.1. echo server

<details>

<summary> 为避免代码之纷繁扰乱加了折叠 </summary>

```C
#include <arpa/inet.h>
#include <netdb.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    // addrinfo: protocol agnostic
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; // ipv4 or ipv6
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE; // assign localhost

    int res = getaddrinfo("127.0.0.1", "3490", &hints, &result);
    if (res != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
        exit(EXIT_FAILURE);
    }

    int sockfd = socket(result->ai_family,
        result->ai_socktype,
        result->ai_protocol);
    if (sockfd == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    res = bind(sockfd, result->ai_addr, result->ai_addrlen);
    if (res != 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

// established yet accepted queue length
#define backlog 8
    res = listen(sockfd, backlog);
    if (res != 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 没有此while，首次client连接断开，server亦会断开
    while (true) {
        struct sockaddr_storage peer;
        socklen_t peer_len = sizeof(struct sockaddr_storage);

        // grab an established connection from kernel accept Q
        int peer_sockfd = accept(sockfd, (struct sockaddr*)&peer, &peer_len);

        if (peer_sockfd < 0) {
            perror("accept");
            // continue;
            exit(EXIT_FAILURE);
        }

        char ipstr[INET6_ADDRSTRLEN];
        void* addr;
        uint16_t port;

        if (peer.ss_family == AF_INET) {
            struct sockaddr_in* ipv4 = (struct sockaddr_in*)&peer;
            addr = &(ipv4->sin_addr);
            // network 大端字节序 -> host 字节序
            port = ntohs(ipv4->sin_port);
        } else {
            struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)&peer;
            addr = &(ipv6->sin6_addr);
            port = ntohs(ipv6->sin6_port);
        }
        inet_ntop(peer.ss_family, addr, ipstr, sizeof(ipstr));
        printf("connection from %s:%d\n", ipstr, port);

        char buffer[15];
        // while 放在 accept 之后
        // 若此 while 包括accpet，会阻塞在第二次 accept，后续不再 recv
        while (true) {
            int recv_cnt = recv(peer_sockfd, buffer, sizeof(buffer), 0);
            if (recv_cnt == -1) {
                perror("recv");
                // 如果 continue，则 client 断开后会无限循环
                break;
            } else if (recv_cnt == 0) // TCP EOF
            {
                printf("connection from %s: %d closed\n", ipstr, port);
                // 如果 continue，则 client 断开后会无限循环
                break;
            } else {
                printf("received %d bytes: ", recv_cnt);

                fflush(stdout);
                write(1, buffer, recv_cnt);
                printf("\n");
            }

            int sent_cnt = send(peer_sockfd, buffer, recv_cnt, 0);
            if (sent_cnt == -1) {
                perror("send back");
                continue;
            } else {
                printf("   sent %d bytes: ", sent_cnt);

                fflush(stdout);
                write(1, buffer, sent_cnt);
                printf("\n");
            }

            // sleep(1);
        }
        close(peer_sockfd);
    }
    close(sockfd);
    freeaddrinfo(result);

    return 0;
}
```

</details>

## 1.2. echo client

<details>

<summary> 为避免代码之纷繁扰乱加了折叠 </summary>

```C
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    int sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int res = getaddrinfo("127.0.0.1", "3490", &hints, &result);

    if (res != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
        exit(EXIT_FAILURE);
    }

    res = connect(sockfd, result->ai_addr, result->ai_addrlen);
    if (res < 0) {
        perror("connect");
        exit(EXIT_FAILURE);
    }
    // fcntl(sockfd, F_SETFL, O_NONBLOCK);

    char buffer[37];
    strcpy(buffer, "123456789ABCDEFG");

    char recv_buf[10];

    while (true) {
        // snprintf(buffer, sizeof(buffer), "%d", pid);
        // strcpy(buffer, "hello word");
        // write data to kernel send buffer
        int sent_cnt = send(sockfd, buffer, strlen(buffer), 0);
        if (sent_cnt == -1) {
            perror("send");
            continue;
        } else {
            printf("\n%d bytes sent: ", sent_cnt);

            fflush(stdout);
            write(1, buffer, sent_cnt);
            printf("\n");
        }

        int recv_cnt = recv(sockfd, recv_buf, sizeof(recv_buf), 0);
        if (recv_cnt == -1) {
            perror("recv");
            continue;
        } else {
            printf("received %d bytes: ", recv_cnt);

            fflush(stdout);
            write(1, recv_buf, recv_cnt);
            printf("\n");
        }
        sleep(1);
    }

    close(sockfd);
    // sleep(999);
    freeaddrinfo(result);

    return 0;
}
```

</details>

# 2. 多进程 echo client-server

代码改动不多，关于多进程需注意

parent: accept + fork

child: recv/send + close(listen_sockfd) + \_exit()

parent: close(peer_sockfd) + continue accept

- 1. fork 即使 CoW 也成本高
  - 1.1. 分配内存空间、创建内核对象、**拷贝页表**、task_struct 等

  - 1.2. 上下文切换：vaddrspace 切换、页表切换（导致 TLB 失效，查询慢速物理页表，性能抖动）

  - 1.3. recv 阻塞不占用 cpu，但是占用 mem / scheduler slot

- 2. 内存独立，需要**进程间通信 IPC**：管道、消息队列、共享内存

- 3. 文件描述符是引用计数，fork 后及时关闭
  - 3.1. parent: close(peer_sockfd)，若不关 child 退出还有引用计数，TCP 不触发 FIN，耗尽 fd
  - 3.2. child: close(listen_sockfd)

- 4. 忽略信号

SIGCHLD：子进程状态变化（terminate、被信号 stop、resume 时，kernel发给父进程的通知信号

child \_exit() -> kernel 标记 zombie -> kernel 给 parent 发 SIGCHLD

parent 收到信号 -> parent waitpid() -> kernel 释放资源 -> zombie 消失

- 5. 使用 **\_exit()**：fork parent 缓冲区，使用 exit() 可能导致已打印内容退出时重复输出

```C
// 关于信号处理部分的代码
void sigchild_handler(int sig)
{
    // prevent waitpid 覆盖主程序的 error
    int saved_errno = errno;

    // 回收所有已退出子进程
    while (waitpid(-1, NULL, WNOHANG) > 0) {
      // WNOHANG 避免阻塞
      // while 循环面向 **signal coalescing**：多 signal 不排队
      // 多进程同时退出，kernel 可能只向 parent 发送一个 SIGCHLD
    }
    errno = saved_errno;
}

int main(){
    struct sigaction sa = { 0 };
    sa.sa_handler = sigchild_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    // SA_RESTART
    // parent 平时阻塞在 accept()，若某 child 退出，kernel 触发 SIGCHLD 。
    // 暂停 accept()，跳转 sigchld_handler，执行完 handler 后回归。
    // 若无 SA_RESTART, accept() 会直接报错返回 -1
    // 并将 errno 设置为 EINTR（Interrupted system call）。
    // 一旦设置 SA_RESTART：kernel 自动重新启动被中断的 accept()

    // SA_NOCLDSTOP
    // client: ctrl-z  SIGSTOP, SIGCONT等

    // SIGCHLD  child 状态变化
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }
}
```

## 2.1. multi-process echo server

<details>

<summary> 为避免代码之纷繁扰乱加了折叠 </summary>

```C
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

void sigchild_handler(int sig)
{
    // prevent waitpid 覆盖主程序的 error
    int saved_errno = errno;

    // sleep(2);
    // 回收所有已退出子进程
    while (waitpid(-1, NULL, WNOHANG) > 0) {
        // 循环因为信号不排队，多进程同时退出，kernel可能只向父进程发送一个 SIGCHLD
    }
    errno = saved_errno;
}

int main()
{
    struct sigaction sa = { 0 };
    sa.sa_handler = sigchild_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    // SA_RESTART
    // parent 平时阻塞在 accept()，若某 child 退出，kernel 触发 SIGCHLD 。
    // 暂停 accept()，跳转 sigchld_handler，执行完 handler 后回归。
    // 若无 SA_RESTART, accept() 会直接报错返回 -1
    // 并将 errno 设置为 EINTR（Interrupted system call）。
    // 一旦设置 SA_RESTART：kernel 自动重新启动被中断的 accept()

    // SA_NOCLDSTOP
    // client: ctrl-z  SIGSTOP, SIGCONT等

    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }

    // addrinfo: protocol agnostic
    struct addrinfo hints,
        *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; // ipv4 or ipv6
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE; // 用于 node = NULL 时，bind any

    int res = getaddrinfo("127.0.0.1", "3490", &hints, &result);
    if (res != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
        exit(EXIT_FAILURE);
    }

    int sockfd = socket(result->ai_family,
        result->ai_socktype,
        result->ai_protocol);
    if (sockfd == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    res = bind(sockfd, result->ai_addr, result->ai_addrlen);
    if (res != 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

// established yet accepted queue length
#define backlog 8
    res = listen(sockfd, backlog);
    if (res != 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 没有此while，首次client连接断开，server亦会断开
    while (true) {
        struct sockaddr_storage peer;
        socklen_t peer_len = sizeof(struct sockaddr_storage);

        // grab an established connection from kernel accept Q
        int peer_sockfd = accept(sockfd, (struct sockaddr*)&peer, &peer_len);
        if (peer_sockfd < 0) {
            perror("accept");
            // continue;
            exit(EXIT_FAILURE);
        }

        pid_t pid = fork();
        if (pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            // child

            // child 不接受新连接
            close(sockfd);

            char ipstr[INET6_ADDRSTRLEN];
            void* addr;
            uint16_t port;

            if (peer.ss_family == AF_INET) {
                struct sockaddr_in* ipv4 = (struct sockaddr_in*)&peer;
                addr = &(ipv4->sin_addr);
                // network 大端字节序 -> host 字节序
                port = ntohs(ipv4->sin_port);
            } else {
                struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)&peer;
                addr = &(ipv6->sin6_addr);
                port = ntohs(ipv6->sin6_port);
            }
            inet_ntop(peer.ss_family, addr, ipstr, sizeof(ipstr));
            printf("server %d: connection from %s:%d\n", getpid(), ipstr, port);

            char buffer[20];
            // while 放在 accept 之后
            // 若此 while 包括accpet，会阻塞在第二次 accept，后续不再 recv
            while (true) {
                int recv_cnt = recv(peer_sockfd, buffer, sizeof(buffer), 0);
                if (recv_cnt == -1) {
                    perror("recv");
                    // 如果 continue，则 client 断开后会无限循环
                    break;
                } else if (recv_cnt == 0) // TCP EOF
                {
                    printf("connection from %s: %d closed\n", ipstr, port);
                    // 如果 continue，则 client 断开后会无限循环
                    break;
                } else {
                    printf("received %d bytes: ", recv_cnt);

                    fflush(stdout);
                    write(1, buffer, recv_cnt);
                    printf("\n");
                }

                int sent_cnt = send(peer_sockfd, buffer, recv_cnt, 0);
                if (sent_cnt == -1) {
                    perror("send back");
                    continue;
                } else {
                    printf("   sent %d bytes: ", sent_cnt);

                    fflush(stdout);
                    write(1, buffer, sent_cnt);
                    printf("\n");
                }
            }

            // sleep(1);
            sleep(5);
            close(peer_sockfd);
            _exit(0);
        } else {
            // parent
            close(peer_sockfd);
        }
    }

    freeaddrinfo(result);

    return 0;
}
```

</details>

## 2.2. multi-process echo client

<details>

<summary> 为避免代码之纷繁扰乱加了折叠 </summary>

```C
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    int sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int res = getaddrinfo("127.0.0.1", "3490", &hints, &result);

    if (res != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(res));
        exit(EXIT_FAILURE);
    }

    res = connect(sockfd, result->ai_addr, result->ai_addrlen);
    if (res < 0) {
        perror("connect");
        exit(EXIT_FAILURE);
    }
    // fcntl(sockfd, F_SETFL, O_NONBLOCK);

    char buffer[37];
    // strcpy(buffer, "%d ABCDEFG");

    snprintf(buffer, sizeof(buffer), "%d: ABCDEFG", getpid());

    char recv_buf[20];

    while (true) {
        // snprintf(buffer, sizeof(buffer), "%d", pid);
        // strcpy(buffer, "hello word");
        // write data to kernel send buffer
        int sent_cnt = send(sockfd, buffer, strlen(buffer), 0);
        if (sent_cnt == -1) {
            perror("send");
            continue;
        } else {
            printf("\n%d bytes sent: ", sent_cnt);

            fflush(stdout);
            write(1, buffer, sent_cnt);
            printf("\n");
        }

        int recv_cnt = recv(sockfd, recv_buf, sizeof(recv_buf), 0);
        if (recv_cnt == -1) {
            perror("recv");
            continue;
        } else {
            printf("received %d bytes: ", recv_cnt);

            fflush(stdout);
            write(1, recv_buf, recv_cnt);
            printf("\n");
        }
        sleep(1);
    }

    close(sockfd);
    // sleep(999);
    freeaddrinfo(result);

    return 0;
}
```

</details>
