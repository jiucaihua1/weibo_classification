# -*- coding: utf-8 -*-
import os

BOT_NAME = 'spider'
LOG_LEVEL = 'INFO'


def _env_int(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default

SPIDER_MODULES = ['spiders']
NEWSPIDER_MODULE = 'spiders'

ROBOTSTXT_OBEY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'cookie.txt'), 'rt', encoding='utf-8') as f:
    cookie = f.read().strip().lstrip('\ufeff')
DEFAULT_REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://weibo.com/',
    'X-Requested-With': 'XMLHttpRequest',
    'Cookie': cookie,
}

CONCURRENT_REQUESTS = _env_int("WEIBO_CONCURRENT_REQUESTS", 4)

DOWNLOAD_DELAY = _env_float("WEIBO_DOWNLOAD_DELAY", 2)
RANDOMIZE_DOWNLOAD_DELAY = True

RETRY_ENABLED = True
RETRY_TIMES = _env_int("WEIBO_RETRY_TIMES", 8)
RETRY_HTTP_CODES = [403, 408, 429, 500, 502, 503, 504, 522, 524]

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': None,
    'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': None,
    'middlewares.IPProxyMiddleware': 100,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 101,
}

ITEM_PIPELINES = {
    'pipelines.JsonWriterPipeline': 300,
}
