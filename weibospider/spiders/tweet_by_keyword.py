#!/usr/bin/env python
# encoding: utf-8
"""
Author: rightyonghu
Created Time: 2022/10/22
"""
import datetime
import json
import re
from scrapy import Spider, Request
from spiders.common import parse_tweet_info
from runtime_config import get_list_config, get_bool_env, get_datetime_env, get_int_env


class TweetSpiderByKeyword(Spider):
    """
    关键词搜索采集
    """
    name = "tweet_spider_by_keyword"
    base_url = "https://s.weibo.com/"

    def start_requests(self):
        """
        爬虫入口
        """
        # 这里keywords可替换成实际待采集的数据
        keywords = get_list_config('WEIBO_KEYWORDS', 'keywords.txt', ['丽江'])
        # 这里的时间可替换成实际需要的时间段
        start_time = get_datetime_env('WEIBO_START_TIME', datetime.datetime(year=2022, month=10, day=1, hour=0))
        end_time = get_datetime_env('WEIBO_END_TIME', datetime.datetime(year=2022, month=10, day=7, hour=23))
        # 是否按照小时进行切分，数据量更大; 对于非热门关键词**不需要**按照小时切分
        is_split_by_hour = get_bool_env('WEIBO_SPLIT_BY_HOUR', default=True)
        max_pages = get_int_env('WEIBO_MAX_PAGES', 0)
        for keyword in keywords:
            if not is_split_by_hour:
                _start_time = start_time.strftime("%Y-%m-%d-%H")
                _end_time = end_time.strftime("%Y-%m-%d-%H")
                url = f"https://s.weibo.com/weibo?q={keyword}&timescope=custom%3A{_start_time}%3A{_end_time}&page=1"
                yield Request(url, callback=self.parse, meta={'keyword': keyword, 'page_num': 1, 'max_pages': max_pages})
            else:
                time_cur = start_time
                while time_cur < end_time:
                    _start_time = time_cur.strftime("%Y-%m-%d-%H")
                    _end_time = (time_cur + datetime.timedelta(hours=1)).strftime("%Y-%m-%d-%H")
                    url = f"https://s.weibo.com/weibo?q={keyword}&timescope=custom%3A{_start_time}%3A{_end_time}&page=1"
                    yield Request(url, callback=self.parse, meta={'keyword': keyword, 'page_num': 1, 'max_pages': max_pages})
                    time_cur = time_cur + datetime.timedelta(hours=1)

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        html = response.text
        if '<p>抱歉，未找到相关结果。</p>' in html:
            self.logger.info(f'no search result. url: {response.url}')
            return
        # Extract bid from result links in a less fragile way.
        tweet_ids = set(re.findall(r'https?://weibo\.com/\d+/([A-Za-z0-9]+)(?:\?|")', html))
        for tweet_id in tweet_ids:
            url = f"https://weibo.com/ajax/statuses/show?id={tweet_id}"
            yield Request(url, callback=self.parse_tweet, meta=response.meta, priority=10)
        next_page = re.search('<a href="(.*?)" class="next">下一页</a>', html)
        if next_page:
            page_num = int(response.meta.get('page_num', 1))
            max_pages = int(response.meta.get('max_pages', 0))
            if max_pages and page_num >= max_pages:
                return
            url = "https://s.weibo.com" + next_page.group(1)
            meta = dict(response.meta)
            meta['page_num'] = page_num + 1
            yield Request(url, callback=self.parse, meta=meta)

    @staticmethod
    def parse_tweet(response):
        """
        解析推文
        """
        data = json.loads(response.text)
        item = parse_tweet_info(data)
        item['keyword'] = response.meta['keyword']
        if item['isLongText']:
            url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
            yield Request(
                url,
                callback=TweetSpiderByKeyword.parse_longtext_with_fallback,
                meta={'item': item, 'handle_httpstatus_list': [403], 'dont_retry': True},
                priority=20,
                errback=TweetSpiderByKeyword.handle_longtext_error
            )
        else:
            yield item

    @staticmethod
    def handle_longtext_error(failure):
        item = failure.request.meta.get('item')
        if item:
            yield item

    @staticmethod
    def parse_longtext_with_fallback(response):
        item = response.meta['item']
        if response.status == 200:
            data = json.loads(response.text).get('data', {})
            if data.get('longTextContent'):
                item['content'] = data['longTextContent']
        yield item
