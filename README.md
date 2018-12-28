* 文書分類サンプルスクリプトおよびデータ (2018/06/29, 中澤 敏明)

** データ

data/train.csv.gz : 訓練データ
data/test.csv.gz  : テストデータ

データはcsv形式のテキストファイルを圧縮したものなので、
gunzip train.csv.gz などとして解凍する


** スクリプト

data.py      : ChainerのDatasetMixinクラスを継承したDatasetクラスの定義
nets.py      : ニューラルネットワークの定義
nlp_utils.py : データをCPUまたはGPUに送るための特別な関数(入力のサイズが固定の場合などは不要)
train.py     : 実際に訓練やテストを行うファイル


** 使い方

- python3 train.py -h でオプションを確認

- 例えば以下のようにすれば、GPUを使った訓練が開始され、結果がresult以下に保存される

python3 --gpu 1 --train_file data/train.seg.csv --test_file data/test.seg.csv
