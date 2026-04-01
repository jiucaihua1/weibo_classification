#!/usr/bin/env python
# encoding: utf-8
import datetime
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")


def _read_list_from_input_file(file_name):
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        return []
    with open(file_path, "rt", encoding="utf-8") as f:
        values = []
        for line in f.readlines():
            value = line.strip().lstrip("\ufeff")
            if value:
                values.append(value)
    return values


def get_list_env(name, default_values=None):
    value = os.getenv(name, "").strip()
    if value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(default_values or [])


def get_bool_env(name, default=False):
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def get_datetime_env(name, default_dt):
    value = os.getenv(name, "").strip()
    if not value:
        return default_dt
    # format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    try:
        if " " in value:
            return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return datetime.datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return default_dt


def get_int_env(name, default=0):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_list_config(env_name, input_file_name, default_values=None):
    values = get_list_env(env_name, [])
    if values:
        return values
    file_values = _read_list_from_input_file(input_file_name)
    if file_values:
        return file_values
    return list(default_values or [])
