# -*- coding: utf-8 -*-
"""
URLを指定してTqdm Progress Barで大きなファイルをダウンロードする
https://medium.com/better-programming/python-progress-bars-with-tqdm-by-example-ce98dbbc9697

Usage:
    $ python download_files.py -u https://nodejs.org/dist/v12.13.1/node-v12.13.1.pkg https://aka.ms/windev_VM_virtualbox # カレントディレクトリに2ファイルDL
    $ python download_files.py -o tmp -u https://nodejs.org/dist/v12.13.1/node-v12.13.1.pkg # tmpディレクトリにDL
"""
#  Copyright 2019 tiptapcode Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import sys
import tqdm
import requests
import validators
import argparse


class FileDownloader(object):

    def get_url_filename(self, url):
        """
        Discover file name from HTTP URL, If none is discovered derive name from http redirect HTTP content header Location
        :param url: Url link to file to download
        :type url: str
        :return: Base filename
        :rtype: str
        """
        try:
            if not validators.url(url):
                raise ValueError('Invalid url')
            filename = os.path.basename(url)
            basename, ext = os.path.splitext(filename)
            if ext:
                return filename
            header = requests.head(url, allow_redirects=False).headers
            return os.path.basename(header.get('Location')) if 'Location' in header else filename
        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
            raise errh
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            raise errc
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
            raise errt
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else", err)
            raise err

    def download_file(self, url, filename=None, target_dir=None):
        """
        Stream downloads files via HTTP
        :param url: Url link to file to download
        :type url: str
        :param filename: filename overrides filename defined in Url param
        :type filename: str
        :param target_dir: target destination directory to download file to
        :type target_dir: str
        :return: Absolute path to target destination where file has been downloaded to
        :rtype: str
        """
        if target_dir and not os.path.isdir(target_dir):
            raise ValueError('Invalid target_dir={} specified'.format(target_dir))
        local_filename = self.get_url_filename(url) if not filename else filename

        req = requests.get(url, stream=True)
        file_size = int(req.headers['Content-Length'])
        chunk_size = 1024  # 1 MB
        num_bars = int(file_size / chunk_size)

        base_path = os.path.abspath(os.path.dirname("__file__"))
        target_dest_dir = os.path.join(base_path, local_filename) if not target_dir else os.path.join(target_dir, local_filename)
        with open(target_dest_dir, 'wb') as fp:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=local_filename, leave=True, file=sys.stdout):
                fp.write(chunk)

        return target_dest_dir


if __name__== "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=None, help="output dir path.")
    ap.add_argument("-u", "--urls", nargs='+', required=True, help="download file urls.")
    args = ap.parse_args()#vars(ap.parse_args())

    #links = ['https://nodejs.org/dist/v12.13.1/node-v12.13.1.pkg', 'https://aka.ms/windev_VM_virtualbox']
    links = args.urls

    os.makedirs(args.output_dir, exist_ok=True)

    downloader = FileDownloader()

    for url in links:
        downloader.download_file(url, target_dir=args.output_dir)