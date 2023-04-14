---
layout: blog
title: Python Crawler Tutorial of BIT
date: 2020-03-13 22:24:36
tags:
---

# 一、Week1:规则

## 1. Library: Requests

### (1). requests
    r = requests.get(url, params=None, **kwargs)

```python
  #!/usr/bin/python
  import requests
  r = request.get('http://www.baidu.com')
  type(r)
  if r.headers == 200:
      print('Access success')
  else if r.headers ==404:
      print('Access denied')

  r.headers
  r.text
  r.encoding            #HTTP header中获取字段
  r.apparent_encoding   #从内容中分析出的响应内容编码方式（备选）
  r.content
```

### (2). universal framework

|||
|:-:|:-:|
|requests.ConnectionError|网络连接错误|
|requests.HTTPError|HTTP错误异常|
|requests.URLRuquired|URL缺失异常|
|requests.TooManyRedirects|重定向|
|requests.ConnectTimeout|连接远程服务器超时异常|
|requests.Timeout|请求URL超时|

```python
#!/usr/bin/python
import requests

def getHTMLtext(url):
    try:
        r = requests.get(url,timeout = 30)
        r.raise_for_status() #如果r.status！= 200，产生异常requests.HTTPError
        r..encoding = r.apparent_encoding
        return r.text
    except:
        return "Error occurred"

    if __name__ == '__main__':
        url = 'https://www.baidu.com'
        print(getHTMLtext)
```

### (3). 7 major methods of Requests

|METHODS|INTRODUCTION|
|:-:|:-:|
|requests.request()||
|requests.get()|major method to get HTML webpage|
|requests.head()|header information|
|requests.post()|submit POST resourses|
|requests.put()|submit PUT(replace)|
|requests.patch()|partially modified requests|
|requests.delete()|delete|

HTTP, Hyper Text Transfer Protocol
URL: `http://host[:port][path]`

```python
#!/usr/bin/python
#requests.post

payload = {'key1':'value1', 'key2':'value2'}
r = requests.post(URL, data = payload')
print(r.text)
'''
{
    ...
    "form":{
        "key2":"value2",
        "key1":"value1"
    },
}
'''

r = requests.post(URL, data = 'just a piece of text')
print(r.text)
'''
{
    ...
    "data":"just a piece of text"
    "form":{}
}
'''

```

### (4). More about Requests

(0). `requests.request(method, url, **kwargs)`  
method = ['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'deleta', 'OPTIONS']  
(1). `params`

```python
#!/usr/bin/python
kv = {'key1':'value1', 'key2':'value2'}
r = requests.request('GET', URL, patams = kv)
print(r.url)
URL?key1=value1&key2=value2
```

(2). `data`

```python
#!/usr/bin/python
kv = {'key1':'value1', 'key2':'value2'}
r = requests.request('POST', URL, data = kv)
body = 'textbody'
r = requests.request('POST', URL, data = body)
```

(3). `json`

```python
kv = {'key1':'value1', 'key2':'value2'}
r = requests.request('POST', URL, json = kv)
```

(4). `headers`

```python
hd = {'user-agent':'Chrome/10'}
r = requests.request('POST', URL, headers = hd)
```

(5). `cookie`  
advanced  

(6). `auth`  
advanced  

(7). `files`  

```python
fs = {'file':open('data.csv', 'rb')}
r = requests.request('POST', URL, files = fs)
```

(8). `timeout`

(9). `proxies`

```python
pxs = {'http':'http://user:pass@10.10.10.1:1234'
       'https':'https://10.10.10.1:4321' }
r = requests.request('GET',URL, proxies = pxs)
```

||**args||
|:-:|:-:|:-:|
|1|params|dictionary or text sequences
|2|data|dictionary, text sequences or file object
|3|json|ad titled|
|4|headers|HTTP headers|
|5|cookies|dict or CookieJar|
|6|auth|tuple|
|7|files|dict, file transfer|
|8|timeout|seconds
|9|proxies|dict
|10|allow_redirects|default: True|
|11|stream|fetch contents, download instantly(default: True)|
|12|verify|verify SSL certificate(default True)
|13|cer|local SSL certificate path|

## 2. 爬虫的‘盗亦有道’

### (1). 爬虫引发的问题

> Requests: 小规模，数据量小，速度不敏感
> Scrapy: 中规模，数据规模较大，速度敏感
> Google, Bing: 大规模，搜索引擎，速度关键

- Sources check：判断HTTP headers的User-Agent域
- Annountment: Robots

### (2). Robots protocol

Robots Exclusion Standard  
location: `host/robots.txt`  

Grammar
> User-agent: *
> Disallow: /

i.e. *https://www.jd.com/robots.txt*
> User-agent: *   
> Disallow: /?*   
> Disallow: /pop/*.html   
> Disallow: /pinpai/*.html?*   
> User-agent: EtaoSpider   
> Disallow: /  
> User-agent: HuihuiSpider   
> Disallow: /   
> User-agent: GwdangSpider   
> Disallow: /   
> User-agent: WochachaSpider   
> Disallow: /  

### (3). Robots protocol

dylan 说：别吃牢饭就好～

### (4). Practice

#### 1. jd.com

京东上默认第一 Lolita 裙的信息（逃）

```python
#!/usr/bin/python
import requests
url = 'https://item.jd.com/55949296412.html'
try:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    print(r.text[:1000])
except:
    print('Error occured')
```

#### 2. Amazon

Amazon上面的 Lolita 裙

```python
import requests
url = 'https://www.amazon.cn/dp/B07MQSJQC4/ref=sr_1_12?__mk_zh_CN=%E4%BA%9A%E9%A9%AC%E9%80%8A%E7%BD%91%E7%AB%99&dchild=1&keywords=lolita&qid=1586515136&sr=8-12'

try:
    kv = {'User-Agent':'Mozilla/5.0'}
    r = requests.request('GET',url,headers = kv)
    r.raise_for_status()
    r.encoding = r.apparent_coding
    print(r.text[1000:2000])
except:
    print('Error occurred')
```

#### 3. baidu关键词提交

`baidu api:` `https://www.baidu.com/s?wd=`+ keywords

```python
import requests

import 
kw = {'wd':'Lolita裙'}
kv = {'User-Agent':'Mozilla/5.0'}
url = 'https://www.baidu.com/s'
try:
    r = requests.get(url, params = kw, headers = kv)
    print(r.request.url)
    r.raise_for_status()
    print(len(r.text))
except:
    print('Error occurred')

```

#### 4. Pictures / Videos

用途：老婆们的图 / 视频

```python
import requests
import os

url = 'https://tva2.sinaimg.cn/large/87c01ec7gy1fsnqqz23i'
root = './'
path = root + url.split('/')[-1]

try:
    if not os.path.exists(path):
        r = requests.request('GET',url)
        r.raise_for_status()
        with open(path,'wb') as f:
            f.write(r.content)
            f.close()
            print('Successfully saved!')
except:
    print('Failed')
```

#### 5. IP address

```python
import requests
url = 'http://m.ip138.com/ip.asp?ip='
ip = input('Enter ur ip address:')
kv = {'User-Agent':'Mozilla/5.0'}
try:
    r = requests.get(url + ip, timeout = 10)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    print(r.text[-500:])
except:
    print('Error occurred')
```

# 二、Week2: 语音系列

## 1. 靓汤 Beautiful Soup

### (1). 靓汤安装

```python
import requests
r = requests.get('http://python123.io/ws/demo.html')
demo = r.text

from bs4 import BeautifulSoup
soup = BeautifulSoup(demo, 'html.parser')
print(soup.prettify())
```

### (2). 靓汤基本元素

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup('<html>data</html>', 'html.parser')
soup2 = BeautifulSoup(open('./demo.html'), 'html.parser')
```

|parser|use|
|:-:|:-:|
|bs4 HTML|BeautifulSoup(mk, 'html.parser')|
|lxml HTML|BeautifulSoup(mk, '`lxml`')|
|lxml XML|BeautifulSoup(mk, '`xml`')|
|html5lib|BeautifulSoup(mk, '`html5lib`')|

|Element|About|
|:-:|:-:|
|Tag|<>和</>标明开头结尾|
|Name|标签名字, `<p`>...`</p`>的名字是 p , 格式：`<tag>`.name|
|Attributes|字典形式， `<tag>`.attrs|
|NavigableString|非属性字符串,<>...</>中字符串， 格式：`<tag>`.string|
|Comment|注释|

```html
 <html>  
   <body>  
       <p class = 'title'>...</p>  
   </body>  
 </html>
```

### (3). HTML内容遍历方法based on bs4

|attrs|about|
|:-:|:-:|
|.contents|`<tag>`子节点存入列表|
|.children|children|
|.descentants|recursive children|

|attrs|about|
|:-:|:-:|
|parent|father
|parents|recursive father

|attrs|about|
|:-:|:-:|
|.next\_sibling|as named|
|.previous\_sibling|as named|
|.next\_siblings|recursively|
|.previous\_siblings|recursively

```python
import requests
r = requests.get('http://python123.io/ws/demo.html')
demo = r.text
demo
```

```html
<html>
    <head>
        <title>This is a python demo page</title>
    </head>
    <body>
        <p class="title">
            <b>The demo python introduces several python courses.</b>
        </p>
        <p class="course">Python is a wonderful general-purpose programming language. You can learn Python from novice to professional by tracking the following courses:
            <a href="http://www.icourse163.org/course/BIT-268001" class="py1" id="link1">Basic Python</a> 
            and<a href="http://www.icourse163.org/course/BIT-1001870001" class="py2" id="link2">Advanced Python</a>.
        </p>
    </body>
</html>
```

```python
#下行遍历
for child in soup.body.children:
    print(child)
```

```python
#上行遍历
for parent in soup.a.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)
```

```python
#平行遍历
for sibling in soup.a.next_siblings:
    print(sibling)
```

### (4). HTML\_encoding based on bs4

```python
print(soup.a.prettify())
```

## 2. 信息提取与标记

XML e**X**tensible **M**arkup **L**anguage  
JSON **J**ava**S**cript **O**bjrct **N**otation  
YAML **Y**AML **A**in't **M**arkup **L**anguage

```
<tag>(..) = <tag>.find_all(..)
i.e.
    soup(..) = soup.find_all(..)
```

### (1). XML, JSON, YAML

```xml
XML
<name>...</name>
<name />
<!--more-->
```

```xml
<person>
    <firstName>forename</firstName>
    <lastName>surname</lastName>
    <address>
        <streetAddr>Happy Road No.2</streetAddr>
        <city>Joyful City</city>
    </address>
    <school>College</school><school>CS</school>
</person>
```

---

```json
JSON
"key":"value"
"key":["value1","value2"]
"key":{
        "subkey":"subvalue" }
```

```json
{
    "firstName":"forename",
    "lastName":"surname",
    "address":{
                "streetAddr":"Happy Road No.2",
                "city":"Joyful City"
                }
    "school":["college","CS"]

}
```

---

```yaml
YAML
key: value
key: #Comment
-value1
-value2
key :
    subkey : subvalue
```

```yaml
firstName: forename
lastName : surname
address:
    streetAddr : Happy Road No.2
    city : Joyful City
school:
- college
- CS
```

### (2). Common way to extract information

```python
for link in soup.find_all('a'):
    print(i.get('href'))
```

### (3). HTML content search based on bs4
soup.find\_all(param)

|param|about|
|:-:|:-:|
|name||
|attrs||
|recursive|default = True|
|string||


|method|about|
|:-:|:-:|
|<>.find||
|<>.find\_parent()||
|<>.find\_parents()||
|<>.find\_next\_sibling()||
|<>.find\_next\_siblings()||
|<>.find\_previous\_sibling()||
|<>.find\_previous\_siblings()||
|<>||

#### (4). Practice

##### 1. University Ranking

```python
import requests
from bs4 import BeautifulSoup
import bs4

def getHTMLText(url):
    try:
        r = requests.request('GET',url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ''

def Univ2List(ulist, html):
    soup = BeautifulSoup(html, 'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].string, tds[1].string, tds[2].string])

def PrintUnivList(ulist, num):
    print('{:^10}\t{:^10}\t{:^10}'.format('排名','学校','分数'))
    for i in range(num):
        u = ulist[i]
        print('{:^10}\t{:^6}\t{:^10}'.format(u[0],u[1],u[2]))

def main():
    ulist = []
    url = 'http://www.zuihaodaxue.com/zuihaodaxuepaiming2019.html'
    html = getHTMLText(url)
    Univ2List(ulist,html)
    PrintUnivList(ulist, 20)

main()

```

Modify the print list.

```python
def PrintUnivList(ulist, num):
    tplt = '{0:^10}\t{1:{3}^10}\t{2:^10}'
    print(tplt.format('排名','学校','分数', chr(12288)))
        for i in range(num):
            u = ulist[i]
            print(tplt.format(u[0],u[1],u[2],chr(12288)))

```

# 三、Week3: MORE

## 1. Regular Expression

### (1). concept

```
i.e.
'PN'
'PYN'
'PYTN'
'PYTHN'
'PYTHON'
=
P(Y|YT|YTH|YTHO)?N
```

```
#i.e. 
'PY'开头
后续存在不多于10个字符
后续字符不能是'P'或'Y'

re: PY[^PY][0:10]
```

### (2). grammar

|操作符|说明|i.e.|
|:-:|:-:|:-:|
|.|any single character||
|[ ]|characters range|[abc]表示a,b,c  [a-z]:a,b...z|
|[^ ]|non-characters|[^abc]表示not a , b or c
|*|前字符0或无穷扩展|abc*表示ab, abc, abcc etc
|+|前字符1或无穷扩展|abc+表示abc, abcc etc
|?|前字符0或1扩展|abc?表示ab, abc
|1|左右任意|abc1def表示abc, def
|{m}|扩展前字符m次|ab{2}c表示abbc
|{m, n}|扩展前字符m-n次|ab{1, 2}c表示abc, abbc
|^|开头|^abc表示abc且在开头
|$|结尾|abc$表示abc且在结尾
|()|分组标记，内部使用1|(abc)表示abc, (abc1def)表示abc, def|
|\d|=[0-9]|
|\w|=[A-Za-z0-9_]|

```
examples

PY[^TH]?ON                  -       PYON, PYaON, PYbON, PYcON
PY{:#}N                     -       PN, PYN, PYYN, PYYN 

```

#### classic examples

|re||
|:-:|:-:|
|^[A-Za-z]+$|字母构成|
|^[A-Za-z0-9]+$|字母和数字|
|^-?\d+$|整数|
|^[0-9]*[1-9][0-9]*$|正整数|
|[1-9]\d{5}|邮政编码|
|[\u4e00-\u9fa5]|中文字符|
|\d{3}-\d{8}1d{4}-\d{7}|国内电话号码|

#### 匹配IP地址的正则表达式

```
0-99     :   [1-9]?\d
100-199  :    1\d{2}
200-249  :    d[0-4]\d
250-255  :    25[0-5]

(( [1-9]?\d | 1\d{2} | 2[0-4]\d | 25[0-5] ).){3}([1-9]?\d | 1\d{2} | 2[0-4]\d | 25[0-5] )

```

### (3). Re

re库采用`raw string`类型表达正则表达式, 表示为`r'text'`
`raw string`不包含转义符的字符串

|function()|about|
|:-:|:-:|
|re.search()|返回match对象|
|re.match()|从string开始位置匹配, 返回match对象|
|re.findall()|搜索字符串，返回列表|
|re.split()|分割字符串,返回列表|
|re.finditer()|返回match对象的迭代元素|
|re.sub()|返回替代后字符串|

#### 1) re.search(pattern, string, flags = 0)

|flags|about|
|:-:|:-:|
|re.I  re.IGNORECASE|忽略大小写，[A-Z]可匹配小写字符|
|re.M  re.MULTILINE|^操作符能将给定字符串每行作为匹配开始|
|re.S  re.DOTALL|.匹配所有字符（默认不包括换行符）|

```python
import re
match = re.search(r'[1-9]\d{5}', 'BIT 100081')
if match:
    print(match.group(0))
```

```
10081
```

#### 2) re.match(pattern, string, flags = 0)

```python
import re
match = re.match(r'[1-9]\d{5}', 'BIT 100081')
if match:
    print(match.group(0))
```

```
None
```

#### 3) re.findall(pattern, string, flags = 0)

```python
import re

```

#### 4) re.split(patter, string, maxsplit = 0, flags = 0)

`maxsplit`最大分割数

```python
import re
re.split(r'[1-9]\d{5}','BIT 100081 TSU 100084', maxsplit = 2)
```

`['BIT', ' TSU', '']`

#### 5) re.finditer(patter, string, flags = 0)

```python
import re
for m in re.finditer(r'[1-9]\d{5}','BIT100081 THU100084'):
    if m:
        print(m.group(0))
```

```
100081
100084
```

#### 6) re.sub(pattern, repl, string, count = 0, flags = 0)

`count`匹配最大次数

```python
re.sub(r'[1-9]\d{5}',':zipcode','BIT100081 THU100084')
```

```
BIT:zipcode THU:zipcode
```


### (4). Re库的match对象

|attr|about|
|:-:|:-:|
|.string|待匹配文本|
|.re|pattern对象（正则表达式）|
|.pos|正则表达式搜索文本的开始位置|
|.endpos|结束位置|

|mathod|about|
|:-:|:-:|
|.group(0)|获得匹配后字符串|
|.start()|匹配字符串在原始字符串的开始位置|
|.end()|匹配字符串在原始字符串的结束位置|
|.span()|返回( .start(), .end() )|

#### regex = re.compile(pattern, flags = 0)

|function()|about|
|:-:|:-:|
|regex.search()|返回match对象|
|regex.match()|从string开始位置匹配, 返回match对象|
|regex.findall()|搜索字符串，返回列表|
|regex.split()|分割字符串,返回列表|
|regex.finditer()|返回match对象的迭代元素|
|regex.sub()|返回替代后字符串|


### (5). Re.贪婪匹配和最小匹配

Re库默认采用`贪婪匹配`, 输出最长字符串

|op|about|
|:-:|:-:|
|*?|前字符扩展0～无限次，最小匹配|
|+?|前字符扩展1～无限次，最小匹配|
|??|前字符扩展0～1次，最小匹配|
|{m, n}?|前字符扩展m～n次，最小匹配|

## 2. 淘宝商品价格比较

```python
import requests
import re

def get_html_text(url):
    try:
        kv = {'user-agent':'Mozilla/5.0', 'cookie':'PLEASE FILL YOUR LOCAL COOKIE'}
        r = requests.request('GET', url, timeout = 30, headers = kv)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ''

def parse_page(ilt, html):
    prices = []
    items = []
    try:
        prices = re.findall(r'"view_price":"[\d\.]+"', html)
        items = re.findall(r'"raw_title":".+?"', html)
        for i in range(len(prices)):
            price = eval(prices[i].split(':')[1])
            item = eval(items[i].split(':')[1])
            ilt.append([price, item])
    except:
        return ''
                                 
def print_goods_list(ilt):
    template = '{:4}\t{:8}\t{:16}'
    print(template.format('num', 'price', 'item'))
    count = 0
    for i in ilt:
        count +=1
        print(template.format(count, i[0], i[1]))

def main():
    goods = 'lolita裙'
    depth = 2
    start_url = 'https://s.taobao.com/search?q=' + goods
    info_list = []
    for i in range(depth):
        try:
            url = start_url + '%s=' + str(44*i)
            html = get_html_text(url)
            parse_page(info_list, html)
        except:
            continue
    print_goods_list(info_list)

main()
```

## 3. 股票数据
