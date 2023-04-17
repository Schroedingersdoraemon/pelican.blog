AUTHOR = 'Big Pineapple'
SITENAME = 'Random Gibberish'
# SITESUBTITLE = 'A random wanderer on Internet'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Asia/Shanghai'

DEFAULT_LANG = 'zh_CN'

# THEME = 'themes/basic'
# THEME = 'themes/simple'
THEME = 'themes/custom'

PLUGINS = [ 'minify',
            'pelican.plugins.render_math']

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

ARTICLE_URL = '{date:%Y}/{slug}.html'
ARTICLE_SAVE_AS = '{date:%Y}/{slug}.html'

# Ignore Git configuration
OUTPUT_RETENTION = ['.git', '.gitignore']

STATIC_PATH = ['files']

# DEFAULT_DATE_FORMAT = '%Y-%m-%d'

# Blogroll
LINKS = (('Gentoo', 'https://gentoo.org'),)

# Social widget
SOCIAL = (('E-mail', 'mailto:dylanturing@protonmail.com'),)

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
