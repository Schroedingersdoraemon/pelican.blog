---
layout: blog
title: Use WeeChat as the IRC client
date: 2021-10-15 22:44:10
tags:
---

# 0. abstract

IRC stands for Internet Relay Client, a chat protocol.

# 1. basic IRC commands

```
/join #channel
/part [#channel] [message]
/nick nickname
/msg nickname message       # private message with no conversation window
/query nickname [message]       # a conversation window till you send message
/quit [message]     #disconnect, message to every channel you are in
```

# 2. irc.libera.chat

| |additional ports|
|:-:|:-:|
|plain-text|6665-6667, 8000-8002|
|TLS|6697, 7000, 7070|

Register your IRC nick:

`/msg NickServ REGISTER YourPassword youremail@example.com`

Then identify

`/msg NickServ IDENTIFY YourNick YourPassword`

To log in

`/connect irc.libera.chat 6667 YourNick:YourPassword`

## 2.1 using SASL

SASL is a method that allows identification to services (NickServ) during the connection process, before anything else happens - therefore eliminating the need to `/msg NickServ identify`.

### 2.1.1 for weechat

much more detailed doc [here](https://libera.chat/guides/sasl)

To use SASL, nickname must be registered.

Set up the connection to Libera.Chat

`/server add libera irc.libera.chat/6697 -ssl`

ensure SSL/TLS is enabled for your connection

```
/set irc.server.libera.addresses "irc.libera.chat/6697"
/set irc.server.libera.ssl on
```

configure SASL

```
/set irc.server.libera.sasl_mechanism PLAIN
/set irc.server.libera.sasl_username <nickname>
/set irc.server.libera.sasl_password <password>
/save
```

### 2.1.2 for Konversation

[see official doc here](https://userbase.kde.org/Konversation/Configuring_SASL_authentication)

## 2.2 using CertFP

[see docs of irc.libera.chat](https://libera.chat/guides/certfp)

## 2.3 Cloaks

- Register and identify.

- `/join #libera-cloak`

- say `!cloakme`

Check cloak status

You should see your cloaked hostmask in /who <yournick>, in /whois <yournick>, in /join or /part messages, and upon connection to the network.
