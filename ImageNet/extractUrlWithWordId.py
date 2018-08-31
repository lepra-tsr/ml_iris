import codecs
import pandas as pd
from urllib import parse as up
import re

# @REFS https://github.com/xkumiyu/imagenet-downloader
#
# ImageNetから、ILSVRC2012で使用された1000クラスに絞ったURLリストを作成する
#
# 1. すべての画像のURLリスト → fall11_urls.txt: tsv['id', 'url']
#  eg.
# n00004475_6590  http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg
#
# 2. コンペで使用された1000クラスのリスト。WordNetに登録された単語のcsvファイル → ILSVRC2012_ClassList.txt: csv[]
#  eg.
# kit fox, Vulpes macrotis
# English setter
# Australian terrier
#
# 3. wordNetIdとクラス名の紐付け(words.txt)
#  eg.
# n00002137   abstraction, abstract entity


def main():
    EXTRACT_ROWS = 3

    # ILSVRC2012で使用されたクラスのうち、先頭EXTRACT_ROWSに対するwordNetIdを抜き出す
    # 、 id, name, label(追加、0～999)に
    words = pd.read_csv('words.txt', header=None, delimiter='\t')
    words.columns = ['id', 'name']
    ILSVRC2012 = pd.read_csv(
        'ILSVRC2012_ClassList.txt',
        header=None,
        delimiter='\t',
        nrows=EXTRACT_ROWS,
    )
    ILSVRC2012.columns = ['name']
    df = pd.merge(words, ILSVRC2012, on='name')
    w_ids = list(df['id'])
    label = pd.DataFrame(df['name'].unique())
    label.columns = ['name']
    label['label'] = label.index
    df = pd.merge(df, label, on='name')

    print('start listing...')
    urls = []
    extractedCount = 0

    with codecs.open('fall11_urls.txt', 'r', 'utf-8', 'ignore') as f:
        for i, l in enumerate(f):
            image_id, url, *tmp = l.strip().split('\t')
            w_id, _ = image_id.split('_')

            if w_id not in w_ids:
                continue

            COMMA_SLASH = '://'
            http, after = url.split(COMMA_SLASH)
            composed_after = up.quote(after)
            composed = COMMA_SLASH.join([http, composed_after])
            if (after != composed_after):
                print(after, composed_after)
            extractedCount += 1
            urls.append({'name': image_id + '.jpg', 'url': composed})

    print('write out to urllist.txt : ', EXTRACT_ROWS,
          ' kinds, ', extractedCount, ' items.')
    pd.DataFrame(urls).to_csv(
        'urllist.txt',
        index=False,
        columns=['name', 'url'],
        header=False,
        sep=' '
    )


if __name__ == '__main__':
    main()
