# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import time
from collections import defaultdict


class JsonWriterPipeline(object):
    """
    写入json文件的pipline
    """

    def __init__(self):
        self.file = None
        self.aggregate_file = None
        self.records_by_user = defaultdict(list)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(os.path.dirname(base_dir), 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.run_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    @staticmethod
    def _extract_user_id(item):
        user = item.get('user') or item.get('comment_user')
        if isinstance(user, dict) and user.get('_id'):
            return str(user['_id'])
        if item.get('user_id'):
            return str(item['user_id'])
        # user_spider output uses `_id` as user id
        if item.get('_id'):
            return str(item['_id'])
        return ''

    @staticmethod
    def _extract_text(item):
        # For user profile items, use description as representative text (optional)
        if item.get('description'):
            return item.get('description', '')
        return item.get('content', '')

    @staticmethod
    def _extract_source_type(spider):
        if spider.name == 'comment':
            return 'comment'
        if spider.name == 'user_spider':
            return 'user'
        return 'tweet'

    def _to_unified_record(self, item, spider):
        user_id = self._extract_user_id(item)
        source_type = self._extract_source_type(spider)
        return {
            "user_id": user_id,
            "text": self._extract_text(item),
            "source_type": source_type,
            "created_at": item.get('created_at'),
            "item_id": str(item.get('_id', '')),
            "spider": spider.name,
            "crawl_time": item.get('crawl_time'),
            "raw": dict(item),
        }

    def process_item(self, item, spider):
        """
        处理item
        """
        if not self.file:
            file_name = f"unified_{self.run_tag}.jsonl"
            output_path = os.path.join(self.output_dir, file_name)
            self.file = open(output_path, 'wt', encoding='utf-8')
        item['crawl_time'] = int(time.time())
        unified_record = self._to_unified_record(item, spider)
        user_id = unified_record["user_id"]
        if user_id:
            self.records_by_user[user_id].append(unified_record)
        line = json.dumps(unified_record, ensure_ascii=False) + "\n"
        self.file.write(line)
        self.file.flush()
        return item

    def close_spider(self, spider):
        # 没有任何 item 时不会打开 unified，此时也不写 user_aggregate，避免大量空文件
        if not self.file:
            return
        self.file.close()
        aggregate_path = os.path.join(self.output_dir, f"user_aggregate_{self.run_tag}.json")
        with open(aggregate_path, 'wt', encoding='utf-8') as aggregate_file:
            json.dump(
                [{"user_id": user_id, "records": records} for user_id, records in self.records_by_user.items()],
                aggregate_file,
                ensure_ascii=False,
                indent=2
            )
