# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Reuires https://github.com/maenu/scholar.py::

    pip install https://github.com/maenu/scholar.py/archive/master.zip

!! Run from interpreter to retain data for debugging access

run find_publications.py with command

"""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
import pickle
import random
import re
import time
from typing import List

from eelbrain._utils import ask
from pybtex.database import BibliographyData, parse_bytes, parse_file
import requests


# Query
API_KEY_PATH = Path('~/.serpapi_key').expanduser()
API_KEY = API_KEY_PATH.read_text().strip()
DIR = Path(__file__).absolute().parent
RAW_SEARCH_CACHE = DIR / 'cache-search.json'
SEARCH_CACHE = DIR / 'cache.json'
SELECTION_PATH = DIR / 'selection.json'
BIBTEX_CACHE = DIR / 'bibtex.pickle'
# Parse
DST = DIR.parent / 'publications.bib'
OBSOLETE = """
Qm4WZWrznhEJ
OdvLIq-CqsUJ
VLeSaj8CLlcJ
ZnxohaAY4fEJ
""".split()
BIORXIV_OBSOLETE = """
10.1101/326785
2018/10/09/439158
10.1101/2020.08.17.254482
10.1101/421727
10.1101/2021.10.15.464457
""".split()
ACRONYMS = ['EEG', 'MEG', 'MRI']


def download_entries() -> List:
    entries = []
    # download entries
    total_results = None
    num = 20
    start = 0
    while True:
        r = requests.get(f"https://serpapi.com/search.json?engine=google_scholar&q=%22eelbrain%22&as_ylo=2015&hl=en&start={start}&num={num}&as_vis=1&api_key={API_KEY}")
        content = json.loads(r.content)
        if not total_results:
            total_results = content['search_information']['total_results']
        entries.extend(content['organic_results'])
        # next page
        start += num
        if start >= total_results:
            break
    return entries


def search(cached: bool):
    # Raw entry list from Google
    if cached:
        with RAW_SEARCH_CACHE.open('rt') as file:
            entries = json.load(file)
    else:
        entries = download_entries()
        with RAW_SEARCH_CACHE.open('wt') as file:
            json.dump(entries, file)
    # One entry per key
    entry_dict = {e['result_id']: e for e in entries}
    if len(entry_dict) != len(entries):
        counter = defaultdict(list)
        for e in entries:
            counter[e['result_id']].append(e)
        duplicates = {k: k_entries for k, k_entries in counter.items() if len(k_entries) > 1}
        print("Warning: duplicate entries for keys\n" + '\n'.join([f" {k}: {len(k_entries)}" for k, k_entries in duplicates.items()]))
        assert not duplicates
    for entry in entry_dict.values():
        entry.pop('position')
        entry.pop('inline_links')
        entry.pop('resources', None)
    entry_dict = {k: entry_dict[k] for k in sorted(entry_dict)}
    with SEARCH_CACHE.open('wt') as file:
        json.dump(entry_dict, file, indent=4)


def select():
    if SELECTION_PATH.exists():
        with SELECTION_PATH.open('rt') as file:
            selection = json.load(file)
    else:
        selection = {}
    with SEARCH_CACHE.open('rt') as file:
        entries = json.load(file)
    # remove entries
    removed = set(selection).difference(entries)
    if removed:
        print(f"Removing entries that are no longer in search results:\n {', '.join(removed)}\n")
        for key in removed:
            del selection[key]
    # ask for each new entry
    for result_id, entry in entries.items():
        if result_id in selection:
            continue
        elif entry['link'].startswith('https://scholar.archive.org'):
            selection[result_id] = False
            continue
        print(entry['title'])
        print(entry['publication_info']['summary'])
        print(entry['link'])
        response = ask("Add?", {'y': 'Include in bibliography', 'n': 'Skip', 'x': 'Abort selection'})
        if response == 'x':
            break
        elif response == 'y':
            selection[result_id] = True
        elif response == 'n':
            selection[result_id] = False
        else:
            raise RuntimeError(f'{response=}')
    with SELECTION_PATH.open('wt') as file:
        json.dump(selection, file, indent=1)


def download():
    "Download BiBTeX entries"
    if BIBTEX_CACHE.exists():
        with BIBTEX_CACHE.open('rb') as file:
            bibtex_cache = pickle.load(file)
    else:
        bibtex_cache = {}
    with SELECTION_PATH.open('rt') as file:
        selections = json.load(file)
    # download missing citations
    new = 0
    for result_id, include in selections.items():
        if not include or result_id in bibtex_cache:
            continue
        r = requests.get(f'https://serpapi.com/search.json?engine=google_scholar_cite&q={result_id}&api_key={API_KEY}')
        content = json.loads(r.content)
        if 'links' not in content:
            print(f"Links not found:\n{content}")
            break
        for style_entry in content['links']:
            if style_entry['name'] == 'BibTeX':
                bibtex_url = style_entry['link']
                break
        else:
            print(f"BibTeX link not found in:\n{r.content.decode()}")
            break

        # wait_hours = 0
        while True:
            r = requests.get(bibtex_url)
            if r.status_code == 403:
                print(f"Error 403:\n{r.content.decode()}\n")
                # wait_hours += 1
                base = 10 * 60 * 60
                sleep_time = random.uniform(base + 5.432, base + 65.476)
                print(f"sleeping for {sleep_time/3600:.1f} hours")
                time.sleep(sleep_time)
                continue
            break
        print(result_id, end=' - ')
        bibtex_cache[result_id] = r.content
        with BIBTEX_CACHE.open('wb') as file:
            pickle.dump(bibtex_cache, file)
        new += 1
        # if new % 2:
        sleep_time = random.uniform(65.432, 605.476)
        # else:
        #     base = 60*60
        #     sleep_time = random.uniform(base + 5.432, base + 335.476)
        print(f"sleeping for {sleep_time/60:.1f} minutes")
        time.sleep(sleep_time)
    if not new:
        print("No new reference")


def parse():
    with BIBTEX_CACHE.open('rb') as file:
        bibtex_entries = pickle.load(file)  # individual BibTeX entries downloaded from scholar
    with SEARCH_CACHE.open('rt') as file:
        raw_entries = json.load(file)  # initial search on scholar
    with SELECTION_PATH.open('rt') as file:
        selection = json.load(file)
    # Load existing citation data
    bib = parse_file(DST, 'bibtex')
    # Remove entries that have been removed upstream
    removed = [key for key, entry in bib.entries.items() if 'google_result_id' in entry.fields and not selection.get(entry.fields['google_result_id'], False)]
    if removed:
        print(f"Removing {', '.join(removed)}")
    # add new
    unseen_keys = set(bib.entries.keys())
    for result_id, raw_bibtex in bibtex_entries.items():
        if not selection.get(result_id, False):
            continue
        elif result_id in OBSOLETE:
            continue
        raw_bibtex = re.sub(rb'[^\x00-\x7F]+', b' ', raw_bibtex)
        # parse entry
        data = parse_bytes(raw_bibtex, 'bibtex')
        assert len(data.entries) == 1
        ((key, entry),) = data.entries.items()
        if key in bib.entries and key in unseen_keys:
            unseen_keys.remove(key)
            continue
        # For later additions, use Google ID
        elif result_id in bib.entries:
            if result_id in unseen_keys:
                unseen_keys.remove(result_id)
                continue
            raise RuntimeError(f"Duplicate key {result_id}")
        url = raw_entries[result_id]['link']
        journal = entry.fields.get('journal', '').lower()
        doi = None
        if entry.type == 'phdthesis':
            pass
        elif journal == 'biorxiv':
            if match := re.match(r"https://www\.biorxiv\.org/content/([\d./]+)\.abstract", url):
                doi = match.group(1)
            elif match := re.match(r"https://www\.biorxiv\.org/content/biorxiv/early/([\d./]+)\.full.pdf", url):
                pass  # DOI unknown
            else:
                raise RuntimeError(f"Can't identify DOI from {url=}")
        elif not journal:
            if entry.fields.get('publisher') == 'PsyArXiv':
                entry.fields['journal'] = '{PsyArXiv}'
            else:
                entry.fields['journal'] = '???'
                print('Warning: missing journal')
        if doi:
            if doi in BIORXIV_OBSOLETE:
                continue
            entry.fields['doi'] = doi
        else:
            entry.fields['url'] = url
        entry.fields['google_result_id'] = result_id
        # fix title
        for repl in ACRONYMS:
            entry.fields['title'] = re.sub(repl, '{' + repl + '}', entry.fields['title'])
        bib.add_entry(result_id, entry)
    # Fix escape-char bug in PybTex
    out = bib.to_bytes('bibtex')
    for bad_char in [br'\&', b'_']:
        out = out.replace(b'\\' + bad_char, bad_char)
    DST.write_bytes(out)


if __name__ == '__main__':
    parser = ArgumentParser(description="Find publications that used Eelbrain on Google Scholar")
    parser.add_argument('task', choices=('search', 'select', 'download', 'parse'))
    parser.add_argument('--cached', default=False, help="When searching, used cached entries")
    args = parser.parse_args()
    if args.task == 'search':
        search(args.cached)
    elif args.task == 'select':
        select()
    elif args.task == 'download':
        download()
    elif args.task == 'parse':
        parse()
