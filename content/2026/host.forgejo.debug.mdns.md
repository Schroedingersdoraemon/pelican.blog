---
title: host forgejo and debug mdns
date: 2026-06-06 18:51
tags: network, mdns
---

[TOC]

---

# 0. intro

使用 roc-rk3328-cc renegade 安装 archlinux arm (alarm) 
并托管了 forgejo 和 syncthing，便于 lan 内多设备同步项目文件

syncthing 很简单，只需要注意一点 **LAN 内取消全局发现**，以减少乱七八糟的中继。

同步路径设为 `$HOME/dev`，使用 syncthing 同步 **dev下不包括 .git 的文件**

另外由于 syncthing 不同步 `.stignore`，
所以让其指向可以同步的 `.stignore-shared`。

```shell
dylan@genteelpc ~/dev $ cat .stignore 
#include .stignore-shared
```

`.stignore-shared` 里面放了一般不要的东西。

```shell
dylan@genteelpc ~/dev $ cat .stignore-shared 
// 忽略常见的依赖库和构建产物
**/node_modules/
**/build/
**/dist/
**/bin/
**/obj/

// 忽略 Python 虚拟环境
**/.venv/
**/venv/
**/ENV/

// 核心：彻底 disable Git
**/.git

// 忽略 IDE 生成的临时配置（建议加上，否则 PC 和 Laptop 的窗口布局会冲突）
**/.vscode/
**/.idea/
**/*.swp
**/*.swo

// 忽略 OS 产生的垃圾文件
**/.DS_Store
**/Thumbs.db%      
```

# 1. forgejo

然后氛围了一个脚本，便于在 dev 下新建项目并将上游指向 `alarm.local`，

私以为这是很不错的折中之法，避免了污染家目录下的 `~/.gitconfig`。

```shell
dylan@genteelpc ~/dev $ cat make-new-project-under-dev.sh 
#!/bin/bash

# ================= 配置区 =================
# 1. Forgejo 的访问地址 (mDNS)
HOST="alarm.local:3000"
# 2. 你的 Forgejo 用户名
USER="..."
# 3. 在网页端生成的 Token (设置 -> 应用 -> 生成令牌)
TOKEN="..."
# 4. 你的本地开发根目录
DEV_DIR="$HOME/dev"
# ==========================================

PROJ_NAME=$1

# 检查是否输入了项目名
if [ -z "$PROJ_NAME" ]; then
    echo "错误: 请输入项目名称"
    echo "用法: ./make-brand-new-proj.sh [项目名]"
    exit 1
fi

TARGET_PATH="$DEV_DIR/$PROJ_NAME"

# 1. 创建并进入目录
if [ ! -d "$TARGET_PATH" ]; then
    mkdir -p "$TARGET_PATH"
    echo "✅ 已创建本地目录: $TARGET_PATH"
fi
cd "$TARGET_PATH"

# 2. 调用 Forgejo API 自动创建远程仓库 (如果不存在)
# 这一步实现了“真·一键”，无需去网页手动创建项目
echo "🚀 正在 Forgejo 上创建远程仓库..."
curl -s -X 'POST' \
  "http://$HOST/api/v1/user/repos" \
  -H "accept: application/json" \
  -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{ \"name\": \"$PROJ_NAME\", \"private\": true, \"auto_init\": false }" > /dev/null

# 3. 初始化本地 Git (如果 .git 不存在)
if [ ! -d ".git" ]; then
    git init
    # 使用 Token 嵌入式 URL，实现免密 Push
    REMOTE_URL="http://$TOKEN@$HOST/$USER/$PROJ_NAME.git"
    git remote add origin "$REMOTE_URL"
    echo "✅ Git 初始化完成，已关联远程地址 (Token 模式)"
else
    echo "ℹ  本地 Git 仓库已存在，跳过初始化。"
fi

# 4. 自动创建第一个提交 (可选，防止初次 push 失败)
if [ ! -f "README.md" ]; then
    echo "# $PROJ_NAME" > README.md
    git add README.md
    git commit -m "Initial commit from script"
    # 第一次推送并建立追踪关系
    git push -u origin $(git symbolic-ref --short HEAD)
    echo "✅ 已完成首次提交并推送到 Forgejo"
fi

echo "------------------------------------------"
echo "项目 '$PROJ_NAME' 准备就绪！"
# echo "本地路径: $TARGET_PATH"
echo "远程地址: http://$HOST/$USER/$PROJ_NAME"
echo "------------------------------------------"
```

Thanks to 多播DNS （multicast DNS, mDNS），可以通过 `alarm.local` 访问 alarm，

which 允许设备在 lan 内通过 `.local` 互相发现，而无需中心化的 DNS server。

`ping alarm.local` 成功，web 访问 `http://alarm.local:3000` 成功。

但是 curl 报错 `Could not resolve host: alarm.local`，于是乎开始 debug

# 1. some tests

## 1.1. alarm

`cat /etc/avahi/avahi-daemon.conf`

检查 `use-ipv4=yes` 和 `use-ipv6=yes`

`avahi-resolve -n alarm.local -4`: 在服务端自测是否能解析出自己的 IPv4 地址。
  
```text
[alarm@alarm ~]$ avahi-resolve -n alarm.local -4
alarm.local     192.168.2.24
```

## 1.2. Gentoo PC

- `ping -4 alarm.local`: 测试 IPv4。

报错 `Address family for hostname not supported`，说明 mDNS 的 IPv4 广播解析失败

- `getent ahosts alarm.local`：看到 ip 说明 nss 通了

- `resolvectl query alarm.local`：查看 `systemd-resolved` 是否能捕捉到多播信号

```text
dylan@genteelpc ~ $ resolvectl query -4 alarm.local
alarm.local: 192.168.2.24                      -- link: enp8s0

-- Information acquired via protocol mDNS/IPv4 in 971us.
-- Data is authenticated: no; Data was acquired via local or encrypted transport: no
-- Data from: cache


dylan@genteelpc ~ $ resolvectl query -6 alarm.local
alarm.local: 240e:359:26d6:4f10:4b3:6eff:fed6:5d6d -- link: enp8s0

-- Information acquired via protocol mDNS/IPv6 in 932us.
-- Data is authenticated: no; Data was acquired via local or encrypted transport: no
-- Data from: cache
```

- **DNS**：`curl -v -4 http://alarm.local:3000`

`* Could not resolve host`：DNS解析失败，可用 `nslookup` 验证
   
`* Trying 192.168.2.24...`：解析成功，但是连接失败。
   
expected output

```text
dylan@genteelpc ~ $ curl -v -I -4 http://alarm.local:3000
* Host alarm.local:3000 was resolved.
* IPv6: (none)
* IPv4: 192.168.2.24
*   Trying 192.168.2.24:3000...
* Established connection to alarm.local (192.168.2.24 port 3000) from 192.168.2.4 port 46924 
* using HTTP/1.x
> HEAD / HTTP/1.1
> Host: alarm.local:3000
> User-Agent: curl/8.19.0
> Accept: */*
> 
* Request completely sent off
< HTTP/1.1 200 OK
HTTP/1.1 200 OK
< Date: Sat, 06 Jun 2026 12:31:17 GMT
Date: Sat, 06 Jun 2026 12:31:17 GMT
< 

* Connection #0 to host alarm.local:3000 left intact
```


---

# 2. 为何 Ping 能通，Curl 报错？


- **Ping (ICMP)**：通常直接调用 syscall `getaddrinfo`。

发现 DNS 解析失败时，会自动触发 `nss-mdns` 多播搜索。

也就是 `/etc/nsswitch.conf` 的相关内容

```shell
dylan@genteelpc ~ $ cat /etc/nsswitch.conf | grep hosts
hosts:      mymachines resolve [!UNAVAIL=return] files myhostname dns
```

- **Curl (HTTP/TCP)**：依赖 `/etc/resolv.conf` 中定义的 DNS 服务器。

  普通文件 `/etc/resolv.conf` 指向路由器，`curl` 询问路由器，后者返回 `NXDOMAIN`.
  
  然后直接报错退出，不再尝试 NSS 的 mDNS。

```text
dylan@genteelpc ~ $ cat /etc/resolv.conf.bak 
# Generated by NetworkManager
nameserver 192.168.2.1
nameserver fe80::3a68:beff:fee1:69d0%enp8s0
```

- **缓存清理**：`sudo resolvectl flush-caches`
      
- **多播路由**：`ip route add 224.0.0.0/4 dev <网卡>`

确保 IPv4 多播流量有明确的出口。

```shell
# ping alarm.local
dylan@genteelpc ~ $ sudo tcpdump -ni any port 5353  
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
21:22:37.742420 dae0  Out IP6 fe80::a8c8:9dff:fea9:3563.5353 > ff02::fb.5353: 0 AAAA (QM)? alarm.local. (29)
21:22:38.948835 dae0  Out IP6 fe80::a8c8:9dff:fea9:3563.5353 > ff02::fb.5353: 0 AAAA (QM)? alarm.local. (29)
21:22:41.198822 dae0  Out IP6 fe80::a8c8:9dff:fea9:3563.5353 > ff02::fb.5353: 0 AAAA (QM)? alarm.local. (29)
21:22:42.734921 enp8s0 M   IP6 fe80::e842:91ff:fe6f:931f.51415 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:22:42.735127 enp8s0 M   IP 192.168.2.8.55773 > 224.0.0.251.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:22:42.735260 enp8s0 M   IP6 240e:359:26d6:4f10:e842:91ff:fe6f:931f.43491 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:22:42.735372 enp8s0 M   IP6 240e:359:26d6:4f10:1ee5:c606:9b97:d7ce.51795 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)


# curl -v -4 -I alarm.local
dylan@genteelpc ~ $ sudo tcpdump -ni any port 5353  
Password: 
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
21:24:02.732433 enp8s0 M   IP6 fe80::e842:91ff:fe6f:931f.51415 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:24:02.732591 enp8s0 M   IP 192.168.2.8.55773 > 224.0.0.251.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:24:02.732737 enp8s0 M   IP6 240e:359:26d6:4f10:e842:91ff:fe6f:931f.43491 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:24:02.732855 enp8s0 M   IP6 240e:359:26d6:4f10:1ee5:c606:9b97:d7ce.51795 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:24:12.724457 enp8s0 M   IP6 240e:359:26d6:4f10:1ee5:c606:9b97:d7ce.51795 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)
21:24:12.724624 enp8s0 M   IP6 fe80::e842:91ff:fe6f:931f.51415 > ff02::fb.5353: 0 PTR (QU)? _lyra-mdns._udp.local. (39)


```


---

# 3. answer： systemd-resolved

原来的 Gentoo with systemd 中，`systemd-resolved` 处理 DNS、mDNS 等任务。

```shell
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
```

DNS 指向 `127.0.0.53`，让所有请求先经过 `resolved` 由其判断
（.local 多播，其余上游）

```shell
# systemd-networkd
dylan@genteelpc ~ $ cat /etc/systemd/network/20-wired.network 
[Match]
Name=enp8s0

[Network]
DHCP=yes
MulticastDNS=yes

# NetworkManager
dylan@genteelpc ~ $ cat /etc/NetworkManager/NetworkManager.conf
[main]
dns=systemd-resolved%
```

**可选**:为了防止 resolve 漏掉某些 v4 记录时，让 avahi 的模块作为 fallback

（之所以可选，因为我测试加不加与否**似乎**都不影响）

```shell
# 这里要加一个 mdns_minimal
dylan@genteelpc ~ $ cat /etc/nsswitch.conf | grep hosts
# Valid databases are: aliases, ethers, group, gshadow, hosts,
hosts:      mymachines resolve [!UNAVAIL=return] mdns_minimal files myhostname dns
```

# 4. nginx reverse proxy

对于 LAN 内访问，使用 nginx 反向代理

先设置一下异常处理,所有服务挂了都跳转到此页面

```nginx
# [alarm@alarm html]$ cat /etc/nginx/error-page.conf 
error_page 404 500 502 503 504 /404.html;

location = /404.html {
    root /usr/share/nginx/html;
    internal;
}
```

然后可以抽象一部分出来，三五个服务感觉换到 caddy 也不是很划算，颇有些目睹二十年屎山
之发展现状.

```nginx
# [alarm@alarm html]$ cat /etc/nginx/proxy-common.conf 
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

proxy_intercept_errors on;
proxy_buffering off;
```

Then comes the `/etc/nginx/nginx.conf`

```nginx
# alarm complains about it
server_names_hash_bucket_size 128;
types_hash_max_size 4096;

# git
server {
    server_name git.alarm.local;
	include error-page.conf;

    location / {
        proxy_pass http://127.0.0.1:3000;
	    include proxy-common.conf;
    }
}

# tools
server {
    server_name tools.alarm.local;
    include error-page.conf;

    location / {
        proxy_pass http://127.0.0.1:8093;
        include proxy-common.conf;
    }
}

# LLM
server {
    server_name llm.alarm.local;
    include error-page.conf;

    location / {
        proxy_pass http://192.168.2.4:12345;
		include proxy-common.conf;
    }
}

# dashdot
server {
    server_name dash.alarm.local;
    include error-page.conf;

    location / {
        proxy_pass http://127.0.0.1:3001;
		include proxy-common.conf;
    }
}


server {
	# 稍微值得注意的就是 IPv6 和 default_server
    listen       80 default_server;
    listen       [::]:80 default_server;
    listen       443 ssl default_server;
    listen       [::]:443 ssl default_server;
    server_name  localhost alarm.local;

    ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;

    location / {
        root   /usr/share/nginx/html;
        index  index.html index.htm;
    }

```

# 5. avahi-publish

为了避免修改路由器或者 LAN 内其他设备，借用 avahi-publish

```shell
[alarm@alarm ~]$ avahi-publish -h
avahi-publish [options] -s <name> <type> <port> [<txt ...>]
avahi-publish [options] -a <host-name> <address>

    -h --help            Show this help
    -V --version         Show version
    -s --service         Publish service
    -a --address         Publish address
    -v --verbose         Enable verbose mode
    -d --domain=DOMAIN   Domain to publish service in
    -H --host=DOMAIN     Host where service resides
       --subtype=SUBTYPE An additional subtype to register this service with
    -R --no-reverse      Do not publish reverse entry with address
    -f --no-fail         Don't fail if the daemon is not available
```


因此期待的指令大约如下

```shell
/usr/bin/avahi-publish -R -a git.alarm.local $IP & 
```

此处借用 hostname （来自 inetutils）

```shell
[alarm@alarm ~]$ hostname --help
Usage: hostname [OPTION...] [NAME]
Show or set the system's host name.

  -a, --aliases              alias names
  -d, --domain               DNS domain name
  -f, --fqdn, --long         DNS host name or FQDN
  -F, --file=FILE            set host name or NIS domain name from FILE
  -i, --ip-addresses         addresses for the host name
  -s, --short                short host name
  -y, --yp, --nis            NIS/YP domain name
  -?, --help                 give this help list
      --usage                give a short usage message
  -V, --version              print program version

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

Report bugs to <bug-inetutils@gnu.org>.
```

最终得到一个很简单的 systemd service

```systemd
# [alarm@alarm ~]$ cat /etc/systemd/system/avahi-alias.service 
[Unit]
Description=Publish Git and Tools Avahi Aliases
After=avahi-daemon.service
Wants=avahi-daemon.service

[Service]
Type=simple
ExecStart=/bin/bash -c ' \
  IP=$(hostname -i); \
  /usr/bin/avahi-publish -R -a git.alarm.local $IP & \
  /usr/bin/avahi-publish -R -a tools.alarm.local $IP & \
  /usr/bin/avahi-publish -R -a dash.alarm.local $IP & \
  /usr/bin/avahi-publish -R -a llm.alarm.local $IP & \
  wait'
Restart=always

[Install]
WantedBy=multi-user.target
```
