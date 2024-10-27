# -*- coding: utf-8 -*-

import datetime
import json
import os
import re
import requests

import configparser
import feedparser
import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from openai import OpenAI
from jinja2 import Template


config = configparser.ConfigParser()
config.read('config.ini')
secs = config.sections()
# Maxnumber of entries to in a feed.xml file
max_entries = 1000


def get_cfg(sec, name, default=None):
    value = config.get(sec, name, fallback=default)
    if value:
        return value.strip('"')


BASE = get_cfg('cfg', 'BASE')
KEYWORD_LENGTH = int(get_cfg('cfg', 'keyword_length'))
SUMMARY_LENGTH = int(get_cfg('cfg', 'summary_length'))
SHORT_SUMMARY_LENGTH = int(get_cfg('cfg', 'short_summary_length'))
LANGUAGE = get_cfg('cfg', 'language')
LENGTH_LOWER_BOUND = 200

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
U_NAME = os.environ.get('U_NAME')
OPENAI_PROXY = os.environ.get('OPENAI_PROXY')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
custom_model = os.environ.get('CUSTOM_MODEL')
deployment_url = f'https://{U_NAME}.github.io/RSS-GPT/'


def fetch_url(url, log_file):
    response = None
    headers = {}
    try:
        ua = UserAgent()
        headers['User-Agent'] = ua.random.strip()
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            with open(log_file, 'a') as f:
                f.write(f"Fetch url error: {response.status_code}\n")
            return None
    except requests.RequestException as e:
        with open(log_file, 'a') as f:
            f.write(f"Fetch url error: {e}\n")
        return None


def fetch_feed(url, log_file):
    feed = None
    response = None
    headers = {}
    try:
        ua = UserAgent()
        headers['User-Agent'] = ua.random.strip()
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            return {'feed': feed, 'status': 'success'}
        elif url.endswith(".xml"):  # OpenAI 的 RSS Feed 地址为 https://openai.com/news/rss.xml，可以直接解析
            feed = feedparser.parse(url)
            return {'feed': feed, 'status': 'success'}
        else:
            with open(log_file, 'a') as f:
                f.write(f"Fetch error: {response.status_code}\n")
            return {'feed': None, 'status': response.status_code}
    except requests.RequestException as e:
        with open(log_file, 'a') as f:
            f.write(f"Fetch error: {e}\n")
        return {'feed': None, 'status': 'failed'}


def generate_untitled(entry):
    try:
        return entry.title
    except:
        try:
            return entry.article[:50]
        except:
            return entry.link


def clean_html(html_content):
    """
    This function is used to clean the HTML content.
    It will remove all the <script>, <style>, <img>, <a>, <video>, <audio>, <iframe>, <input> tags.
    Returns:
        Cleaned text for summarization
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for script in soup.find_all("script"):
        script.decompose()

    for style in soup.find_all("style"):
        style.decompose()

    image_urls = []
    for img in soup.find_all("img"):
        img.decompose()
        # if "src" in img.attrs:
        #     image_urls.append(img["src"])
        #     print(image_urls)
        # else:
        #     img.decompose()

    for a in soup.find_all("a"):
        a.decompose()

    for video in soup.find_all("video"):
        video.decompose()

    for audio in soup.find_all("audio"):
        audio.decompose()

    for iframe in soup.find_all("iframe"):
        iframe.decompose()

    for input in soup.find_all("input"):
        input.decompose()

    feed_content = soup.get_text()

    return feed_content, image_urls


def filter_entry(entry, filter_apply, filter_type, filter_rule):
    """
    This function is used to filter the RSS feed.

    Args:
        entry: RSS feed entry
        filter_apply: title, article or link
        filter_type: include or exclude or regex match or regex not match
        filter_rule: regex rule or keyword rule, depends on the filter_type

    Raises:
        Exception: filter_apply not supported
        Exception: filter_type not supported
    """
    if filter_apply == 'title':
        text = entry.title
    elif filter_apply == 'article':
        text = entry.article
    elif filter_apply == 'link':
        text = entry.link
    elif not filter_apply:
        return True
    else:
        raise Exception('filter_apply not supported')

    if filter_type == 'include':
        return re.search(filter_rule, text)
    elif filter_type == 'exclude':
        return not re.search(filter_rule, text)
    elif filter_type == 'regex match':
        return re.search(filter_rule, text)
    elif filter_type == 'regex not match':
        return not re.search(filter_rule, text)
    elif not filter_type:
        return True
    else:
        raise Exception('filter_type not supported')


def read_entry_from_file(sec):
    """
    This function is used to read the RSS feed entries from the feed.xml file.

    Args:
        sec: section name in config.ini
    """
    out_dir = os.path.join(BASE, get_cfg(sec, 'name'))
    try:
        with open(out_dir + '.xml', 'r') as f:
            rss = f.read()
        feed = feedparser.parse(rss)
        return feed.entries
    except Exception:
        return []


def truncate_entries(entries, max_entries):
    if len(entries) > max_entries:
        entries = entries[:max_entries]
    return entries


def gpt_summary(query, image_urls, model, language):
    content = [
        {"type": "text", "text": query}
    ]

    # for image_url in image_urls:
    #     content.append(
    #         {
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": image_url,
    #                 "detail": "low",
    #             },
    #         }
    #     )

    prompt = ""
    with open("prompt.txt", "r", encoding='utf-8') as f:
        prompt = f.read().format(
            short_summary_length=SHORT_SUMMARY_LENGTH,
            summary_length=SUMMARY_LENGTH,
            keyword_length=KEYWORD_LENGTH
        )

    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": content
        },
    ]

    if not OPENAI_PROXY:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    else:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            http_client=httpx.Client(proxy=OPENAI_PROXY),
        )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
    )

    match = re.search(r'```json\n(.*?)```', completion.choices[0].message.content, re.DOTALL)
    json_str = match.group(1).strip()
    response = json.loads(json_str)

    return response


def process(sec, language):
    """ output
    This function is used to output the summary of the RSS feed.

    Args:
        sec: section name in config.ini

    Raises:
        Exception: filter_apply, type, rule must be set together in config.ini
    """
    log_file = os.path.join(BASE, get_cfg(sec, 'name') + '.log')
    out_dir = os.path.join(BASE, get_cfg(sec, 'name'))
    # read rss_url as a list separated by comma
    rss_urls = get_cfg(sec, 'url')
    rss_urls = rss_urls.split(',')

    # RSS feed filter apply, filter title, article or link, summarize title, article or link
    filter_apply = get_cfg(sec, 'filter_apply')

    # RSS feed filter type, include or exclude or regex match or regex not match
    filter_type = get_cfg(sec, 'filter_type')

    # Regex rule or keyword rule, depends on the filter_type
    filter_rule = get_cfg(sec, 'filter_rule')

    # filter_apply, type, rule must be set together
    if filter_apply and filter_type and filter_rule:
        pass
    elif not filter_apply and not filter_type and not filter_rule:
        pass
    else:
        raise Exception('filter_apply, type, rule must be set together')

    # Max number of items to summarize
    max_items = get_cfg(sec, 'max_items')
    if not max_items:
        max_items = 0
    else:
        max_items = int(max_items)
    cnt = 0
    existing_entries = read_entry_from_file(sec)
    with open(log_file, 'a') as f:
        f.write('------------------------------------------------------\n')
        f.write(f'Started: {datetime.datetime.now()}\n')
        f.write(f'Existing_entries: {len(existing_entries)}\n')
    existing_entries = truncate_entries(existing_entries, max_entries=max_entries)
    # Be careful when the deleted ones are still in the feed, in that case, you will mess up the order of the entries.
    # Truncating old entries is for limiting the file size, 1000 is a safe number to avoid messing up the order.
    append_entries = []

    for rss_url in rss_urls:
        with open(log_file, 'a') as f:
            f.write(f"Fetching from {rss_url}\n")
            print(f"Fetching from {rss_url}")
        feed = fetch_feed(rss_url, log_file)['feed']
        if not feed:
            with open(log_file, 'a') as f:
                f.write(f"Fetch failed from {rss_url}\n")
            continue
        for entry in feed.entries:
            if cnt > max_entries:
                with open(log_file, 'a') as f:
                    f.write(f"Skip from: [{entry.title}]({entry.link})\n")
                break

            if entry.link.find('#replay') and entry.link.find('v2ex'):
                entry.link = entry.link.split('#')[0]

            if entry.link in [x.link for x in existing_entries]:
                continue

            if entry.link in [x.link for x in append_entries]:
                continue

            entry.title = generate_untitled(entry)

            try:
                entry.article = entry.content[0].value
            except Exception:
                try:
                    entry.article = entry.description
                except Exception:
                    entry.article = entry.title

            cleaned_article, image_urls = clean_html(entry.article)
            # 如果 RSS Feed 文章太短，直接从链接拉取原文
            if (len(cleaned_article) < LENGTH_LOWER_BOUND):
                original_content = fetch_url(entry.link, log_file)

                if original_content:
                    cleaned_article, image_urls = clean_html(original_content)

            if not filter_entry(entry, filter_apply, filter_type, filter_rule):
                with open(log_file, 'a') as f:
                    f.write(f"Filter: [{entry.title}]({entry.link})\n")
                continue

            gpt_response = {}
            token_length = len(cleaned_article)
            cnt += 1
            if cnt > max_items:
                entry.summary = None
            elif OPENAI_API_KEY and (token_length > LENGTH_LOWER_BOUND):
                if custom_model:
                    try:
                        gpt_response = gpt_summary(cleaned_article, image_urls, model=custom_model, language=LANGUAGE)
                        with open(log_file, 'a') as f:
                            f.write(f"Token length: {token_length}\n")
                            f.write(f"Summarized using {custom_model}\n")
                    except Exception as e:
                        entry.summary = None
                        with open(log_file, 'a') as f:
                            f.write("Summarization failed, append the original article\n")
                            f.write(f"error: {e}. Line: {e.__traceback__.tb_lineno}.\n")
                else:
                    try:
                        gpt_response = gpt_summary(cleaned_article, image_urls, model="gpt-4o-mini", language=LANGUAGE)
                        with open(log_file, 'a') as f:
                            f.write(f"Token length: {token_length}\n")
                            f.write("Summarized using gpt-4o-mini\n")
                    except Exception:
                        try:
                            gpt_response = gpt_summary(cleaned_article, image_urls,
                                                       model="gpt-4-turbo-preview", language=LANGUAGE)
                            with open(log_file, 'a') as f:
                                f.write(f"Token length: {token_length}\n")
                                f.write("Summarized using GPT-4-turbo-preview\n")
                        except Exception as e:
                            entry.summary = None
                            with open(log_file, 'a') as f:
                                f.write("Summarization failed, append the original article\n")
                                f.write(f"error: {e}. Line: {e.__traceback__.tb_lineno}.\n")

            if gpt_response.get("title", ""):
                entry.title = gpt_response["title"]
            if gpt_response.get("short_summary", "") and gpt_response.get("summary", ""):
                short_summary_color = "color:gray;"
                entry.summary = (f'<p style={short_summary_color}>{gpt_response["short_summary"]}</p><br><br>' +
                                 f'<p><strong>摘要：</strong> {gpt_response["summary"]}</p><br><br>' +
                                 f'<p><em>使用 {custom_model} 生成 </em></p>' +
                                 f'<a href={entry.link} target="_blank">查看原文</a>')

            if gpt_response.get("keyword", "") != "ADs":
                append_entries.append(entry)
            with open(log_file, 'a') as f:
                f.write(f"Append: [{entry.title}]({entry.link})\n")

    with open(log_file, 'a') as f:
        f.write(f'append_entries: {len(append_entries)}\n')

    template = Template(open('template.xml').read())

    try:
        rss = template.render(feed=feed, append_entries=append_entries, existing_entries=existing_entries)
        with open(out_dir + '.xml', 'w') as f:
            f.write(rss)
        with open(log_file, 'a') as f:
            f.write(f'Finish: {datetime.datetime.now()}\n')
    except Exception:
        with open(log_file, 'a') as f:
            f.write(f"error when rendering xml, skip {out_dir}\n")
            print(f"error when rendering xml, skip {out_dir}\n")


try:
    os.mkdir(BASE)
except Exception:
    pass

feeds = []
links = []

for x in secs[1:]:
    process(x, language=LANGUAGE)
    feed = {"url": get_cfg(x, 'url').replace(',', '<br>'), "name": get_cfg(x, 'name')}
    feeds.append(feed)  # for rendering index.html
    links.append("- " + get_cfg(x, 'url').replace(',', ', ') + " -> " + deployment_url + feed['name'] + ".xml\n")


def append_readme(readme, links):
    with open(readme, 'r') as f:
        readme_lines = f.readlines()
    while readme_lines[-1].startswith('- ') or readme_lines[-1] == '\n':
        readme_lines = readme_lines[:-1]  # remove 1 line from the end for each feed
    readme_lines.append('\n')
    readme_lines.extend(links)
    with open(readme, 'w') as f:
        f.writelines(readme_lines)


append_readme("README.md", links)
append_readme("README-zh.md", links)

# Rendering index.html used in my GitHub page, delete this if you don't need it.
# Modify template.html to change the style
with open(os.path.join(BASE, 'index.html'), 'w') as f:
    template = Template(open('template.html').read())
    html = template.render(update_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), feeds=feeds)
    f.write(html)
