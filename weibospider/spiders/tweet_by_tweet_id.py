#!/usr/bin/env python
# encoding: utf-8
"""
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2020/4/14
"""
import json
from scrapy import Spider
from scrapy.http import Request
from spiders.common import parse_tweet_info
from runtime_config import get_list_config


class TweetSpiderByTweetID(Spider):
    """
    用户推文ID采集推文
    """
    name = "tweet_spider_by_tweet_id"
    base_url = "https://weibo.cn"

    def start_requests(self):
        """
        爬虫入口
        """
        # 这里user_ids可替换成实际待采集的数据
        tweet_ids = get_list_config('WEIBO_TWEET_IDS', 'tweet_ids.txt', ['LqlZNhJFm'])
        for tweet_id in tweet_ids:
            url = f"https://weibo.com/ajax/statuses/show?id={tweet_id}"
            yield Request(url, callback=self.parse)

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        data = json.loads(response.text)
        item = parse_tweet_info(data)
        if item['isLongText']:
            url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
            yield Request(
                url,
                callback=self.parse_longtext_with_fallback,
                meta={'item': item, 'handle_httpstatus_list': [403], 'dont_retry': True},
                errback=self.handle_longtext_error
            )
        else:
            yield item

    def handle_longtext_error(self, failure):
        item = failure.request.meta.get('item')
        if item:
            self.logger.warning(f"longtext failed, fallback to text_raw: {item.get('mblogid', '')}")
            yield item

    def parse_longtext_with_fallback(self, response):
        item = response.meta['item']
        if response.status == 200:
            data = json.loads(response.text).get('data', {})
            if data.get('longTextContent'):
                item['content'] = data['longTextContent']
        else:
            self.logger.warning(f"longtext {response.status}, fallback to text_raw: {item.get('mblogid', '')}")
        yield item
