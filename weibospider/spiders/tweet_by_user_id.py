#!/usr/bin/env python
# encoding: utf-8
"""
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2020/4/14
"""
import datetime
import json
import re

from scrapy import Spider
from scrapy.http import Request
from spiders.common import parse_tweet_info
from runtime_config import get_list_config, get_bool_env, get_datetime_env, get_int_env


class TweetSpiderByUserID(Spider):
    """
    用户推文数据采集
    """
    name = "tweet_spider_by_user_id"

    def start_requests(self):
        """
        爬虫入口
        """
        # 这里user_ids可替换成实际待采集的数据
        user_ids = get_list_config('WEIBO_USER_IDS', 'user_ids.txt', ['1087770692'])
        # 这里的时间替换成实际需要的时间段，如果要采集用户全部推文 is_crawl_specific_time_span 设置为False
        is_crawl_specific_time_span = get_bool_env('WEIBO_CRAWL_TIME_SPAN', default=False)
        start_time = get_datetime_env('WEIBO_START_TIME', datetime.datetime(year=2022, month=1, day=1))
        end_time = get_datetime_env('WEIBO_END_TIME', datetime.datetime(year=2023, month=1, day=1))
        max_pages = get_int_env('WEIBO_MAX_PAGES', 0)
        for user_id in user_ids:
            url = f"https://weibo.com/ajax/statuses/searchProfile?uid={user_id}&page=1&hasori=1&hastext=1&haspic=1&hasvideo=1&hasmusic=1&hasret=1"
            if not is_crawl_specific_time_span:
                yield Request(url, callback=self.parse, meta={'user_id': user_id, 'page_num': 1, 'max_pages': max_pages})
            else:
                # 切分成10天进行
                tmp_start_time = start_time
                while tmp_start_time <= end_time:
                    tmp_end_time = tmp_start_time + datetime.timedelta(days=10)
                    tmp_end_time = min(tmp_end_time, end_time)
                    tmp_url = url + f'&starttime={int(tmp_start_time.timestamp())}&endtime={int(tmp_end_time.timestamp())}'
                    yield Request(
                        tmp_url,
                        callback=self.parse,
                        meta={'user_id': user_id, 'page_num': 1, 'max_pages': max_pages},
                    )
                    tmp_start_time = tmp_end_time + datetime.timedelta(days=1)

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        try:
            data = json.loads(response.text)
        except Exception:
            self.logger.warning(f"invalid json in timeline response, skip: {response.url}")
            return
        tweets = ((data or {}).get('data') or {}).get('list') or []
        if not isinstance(tweets, list):
            self.logger.warning(f"unexpected timeline payload shape, skip: {response.url}")
            return
        for tweet in tweets:
            try:
                item = parse_tweet_info(tweet)
            except KeyError as exc:
                self.logger.warning(
                    f"skip malformed tweet payload missing key={exc} url={response.url}"
                )
                continue
            except Exception:
                self.logger.warning(f"skip malformed tweet payload url={response.url}")
                continue
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
        if tweets:
            user_id, page_num = response.meta['user_id'], response.meta['page_num']
            max_pages = response.meta.get('max_pages', 0)
            if max_pages and page_num >= max_pages:
                return
            url = response.url.replace(f'page={page_num}', f'page={page_num + 1}')
            yield Request(
                url,
                callback=self.parse,
                meta={'user_id': user_id, 'page_num': page_num + 1, 'max_pages': max_pages},
            )

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
