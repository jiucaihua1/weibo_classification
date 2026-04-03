"""
根据簇关键词与预设「兴趣抽屉」交集数量，为每个簇自动命名。
与业务约定的 category_dict 保持一致，可随时扩展抽屉词表。
"""

from __future__ import annotations

# 兴趣标签抽屉（可随时往里面加词）
CATEGORY_DRAWER: dict[str, list[str]] = {
    "科技数码": [
        "ai",
        "华为",
        "手机",
        "万兴",
        "科技",
        "数码",
        "芯片",
        "评测",
        "小米",
        "苹果",
    ],
    "生活摄影": [
        "摄影",
        "生活",
        "晚安",
        "早安",
        "美食",
        "手记",
        "日常",
        "打卡",
        "风景",
        "旅游",
    ],
    "时政": [
        "伊朗",
        "美国",
        "中国",
        "特朗普",
        "汽车",
        "张雪",
        "国际",
        "政治",
        "底盘",
        "空悬",
        "车展",
    ],
    "日常": [
        "哈哈哈",
        "啊啊啊",
        "没有",
        "真的",
        "现在",
        "时候",
        "什么",
        "喜欢",
        "无语",
        "开心",
        "无聊",
    ],
    "饭圈娱乐": [
        "丁禹",
        "王源",
        "赵皑",
        "陈伦",
        "游戏",
        "天才",
        "海虾",
        "冬季",
        "宫令",
        "演唱会",
        "打榜",
        "粉丝",
    ],
}

UNKNOWN_CLUSTER_LABEL = "未知/混合兴趣"


def auto_name_cluster(
    cluster_keywords: list[str],
    category_dict: dict[str, list[str]] | None = None,
) -> str:
    """命中同一抽屉的词最多则取该抽屉名；否则为未知/混合兴趣。"""
    drawer = category_dict if category_dict is not None else CATEGORY_DRAWER
    if not cluster_keywords:
        return UNKNOWN_CLUSTER_LABEL
    kw_set = set(cluster_keywords)
    best_label = UNKNOWN_CLUSTER_LABEL
    max_matches = 0
    for label, dict_words in drawer.items():
        matches = len(kw_set & set(dict_words))
        if matches > max_matches:
            max_matches = matches
            best_label = label
    return best_label


def map_cluster_keywords_to_names(
    cluster_keywords_by_id: dict[str, list[str]],
    category_dict: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """cluster_keywords meta 形态: {\"0\": [\"ai\", ...], ...} → {\"0\": \"科技数码\", ...}"""
    out: dict[str, str] = {}
    for cid, words in cluster_keywords_by_id.items():
        if not isinstance(words, list):
            continue
        out[str(cid)] = auto_name_cluster(words, category_dict)
    return out
