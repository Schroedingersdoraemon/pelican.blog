---
title: Hosting your dedicated VPN on your own server
date: 2022-04-28 23:16:52
tags:
---

# 1. Prerequisite

\>=net-proxy/shadowsocks-libev-3.3.1

# 2. Configuration

The configuration file is usually located at `/etc/shadowsocks-libev/config.json`

```json
{
	# define lines starting with # as comments
	"server":["::0","0,0,0,0"],

	"server_port":8388,

	# the following method is one of `AEAD ciphers`, which could resist probing from the GFW
	# which includes `chacha20-ietf-poly1305`, `aes-256-gcm`, `aes-128-gcm`
	"method":"chacha20-ietf-poly1305",


	# a much stronger password should be used
	# you coulde use `openssl rand -base64 16` to generate a random password
	"password":"*YourPassword*",

	"mode":"tcp_and_udp",

	"fast_open":false }
```

# 3. Fire Wall

```
# ufw allow ssh
# ufw allow 8388
# ufw enable
```

# 4. Run

```bash
# start and enable the server daemon for systemd
sudo systemctl start shadowsocks-libev-server@.service
sudo systemctl enable shadowsocks-libev-server@.service

# start and enable the server daemon for openrc
sudo rc-service shadowsocks.server start
sudo rc-update add shadowsocks.server
```

# 5. In case ...

Redirect received traffic of both TCP and UDP ranges between 8389 and 8399 to 8388

```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 8389:8399 -j REDIRECT --to-port 8388
sudo iptables -t nat -A PREROUTING -p udp --dport 8389:8399 -j REDIRECT --to-port 8388
```
