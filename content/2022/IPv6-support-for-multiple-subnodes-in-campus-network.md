---
title: IPv6 support for multiple subnodes in campus network
date: 2022-07-29 18:41:05
tags:
---

# 1. WAN

IPv6 connection type: Native DHCP v6

Acquire IPv6 WAN type: Stateless: RA

DNSv6 WAN Settings:

- 240c::6666
- 2400:3200::1
- 2001:4860:4860::8888

IPv6 LAN Settings:

- fc00:101:101::1

IPv6 LAN Prefix Length: 64

LAN DHCPv6 Server: Stateless

# 2. scripts

Customization -> Scripts -> After Firewall rules

```bash
ip6tables -F
ip6tables -P INPUT ACCEPT
ip6tables -P FORWARD ACCEPT
ip6tables -P OUTPUT ACCEPT
```
