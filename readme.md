# Gunosyニュース記事分類モジュール

以下のカスタムコマンドが用意されている
- `scrape`: [Gunosy](https://gunosy.com/)から現在掲載されている全カテゴリの全記事を抽出しデータセットの格納する。
- `train`: データベース上にあるデータを使ってNaiveBayesをトレーニングする。scrapeを実行した後でなければ実行できない。
- `evaluate`: データベース上にあるデータを使ってNaiveBayesモデルをトレーニングし精度を評価する（モデルは直ちに破棄される）。
- `runserver`: テキスト分類用のサーバーが立ち上がる。デフォルトではlocalhost:8000/classifierにローカルサーバーが立ち上がり分類したいGunosy記事の入力を受け付ける。デフォルトではBFC(後述)を利用する設定になっているのでscrapeおよびtrain_bfcコマンド実行後でなければ実行できない。
- `train_bfc`: データベース上にあるデータを使ってBFC(後述)をトレーニングする。scrapeを実行した後でなければ実行できない。(メモリエラーは./classifier/__init__.py内Evaluation.MAXIMUM_DATA_SIZEを調節することで回避）
- `evaluate_bfc`: データベース上にあるデータを使ってBFC（後述）をトレーニングし精度を評価する（モデルは直ちに破棄される）。


# インストール方法

**プロジェクトルート**で以下のコマンドを実行すると実行環境がビルドされたDockerImageがビルドされる

```docker build -t image_name -f ./docker/Dockerfile .```

例えば以下のようなコマンドで上記のカスタムコマンドを受け付けることができる。

```
docker run -it --rm -v volume_name:/GunosyClassifier/data image_name bash
cd GunosyClassifier
python manage.py custom_command_name
```

# テキスト分類手法
## NaiveBayesClassifier
NaiveBayesによるテキスト分類を行う。学習プロセスの概要は以下。
1. まずテキストをBagOfWords特徴量に変換する。すなわちある単語iがテキスト内に登場する場合i番目の要素は１であり、登場しない場合はi番目の要素が０であるようなベクトルを作成する。記事中に同じ単語が複数並んでも対応する特徴量は１である。ただし慣例にならって日本語はMeCabによって形態素解析を行い名詞のみを抽出する。
2. ストップワードを削除する。（Optional. 精度への寄与が見られなかったのでデフォルトでは無効。train, evaluateに`-s`オプションを付加すると得られる。)
3. 特徴量をもとにNaiveBayesをトレーニングする(TensorFlowで実装)。

## BertFeatureClassifier
日本語データセットでトレーニング済の[BERT by huggingface](https://huggingface.co/transformers/pretrained_models.html)の中間層を用いてテキストの特徴量を抽出し、この特徴量の上でRandomForestを用いたテキスト分類器を作成した。
潜在的にはRandomForestに限らずSVMなどの任意の古典的な分類器が利用可能なのでBertFeatureClassifierと命名。SVMは精度がRandomForestに劣ったので利用しなかった。

学習プロセスの概要は以下。
1. テキストを学習済みBERTモデルによって768次元の特徴ベクトルに変換する
2. この上で任意の古典的分類器を使って学習する。本ソフトウェアはRandomForestを用いる。


# 精度評価
２つの手法についてAccuracy, Confusion Matrixによる評価を行った。
評価はscrapeコマンドを１度実行して得た3958個の記事を8:2の比率でトレーニングデータ、テストデータに分割して行った。

以下にはその結果を示す。


## NaiveBayesClassifier
```python manage.py evaluate```

accuracy: 0.7243994943109987

confusion_matrix: 

```math
[[144   9   0   0   0   0   1   0]
 [  0 219   0   0   0   0   0   0]
 [ 17  16  26   0   1   0   1   0]
 [  6  38   0  17   0   0   3   0]
 [  3  29   0   0  11   0   1   0]
 [ 22   7   0   0   0  41   1   0]
 [ 14  17   1   0   0   2  68   0]
 [ 25   3   0   0   0   0   1  47]]
```

## BertFeatureClassifier
```python manage.py evaluate_bfc```

accuracy: 0.8394437420986094

confusion_matrix: 

```math
[[143   7   0   1   0   4   4   1]
 [  7 222   0   0   0   0   0   0]
 [  5   1  45   0   0   5   2   1]
 [  6  12   0  31   1   0   8   0]
 [  8   3   0   5  16   0   1   0]
 [  5   0   3   0   0  63   1   4]
 [ 15   0   0   3   0   2  79   0]
 [  9   2   0   0   0   0   1  65]]
```

## BertFeatureClassifier(scrape ×2)
用いる記事を7910個に増量した場合のBertFeatureClassifierの精度も示す。
評価は上に習い8:2の比率でトレーニングデータ、テストデータに分割して行った。

```python manage.py evaluate_bfc```

accuracy: 0.8805

confusion_matrix: 

```math
[[328   7   3   3   1   2   4   2]
 [  5 425   0   2   0   1   0   0]
 [  8   0  97   0   0   7   1   2]
 [  6  16   0  67   8   6  14   0]
 [ 12   4   0  10  43   0   5   0]
 [  8   1   0   2   1 126   3   4]
 [ 11   0   4   2   1   2 161   2]
 [ 10   1   0   0   0   4   4 146]]
 ```

 ## 考察

### 不均衡性の影響
NaiveBayesClassifierは不均衡なデータの特徴を正確に捉えることができていないことが見て取れる。カテゴリ０（エンタメ）やカテゴリ１（スポーツ）はそもそも記事の数が多く、誤分類先のクラスになってしまうことが多い。

一方BFCは学習器は単なるRandomForestだが特徴量の生成には日本語Wikiなどで学習を終えたBERTを用いている。NaiveBayesに比べて遥かに高い分類精度を誇り、データの不均衡さの影響も抑えることができている。

### BertFeatureClassifierの有用性
BERTは事前に巨大なデータセット上でtrainされた強力な自然言語解析用のニューラルネットワークである。後続のGPT-3は非常に高い文章生成能力で世間を騒がせている。

しかしながらこれらのニューラルネットをトレーニングすることは必ずしも簡単ではない。学習済みモデルを特定のタスクへ転移学習することは多くの計算リソースを要する。

タスクやデータ・セットによってはSVMやRandomForestなどの古典的な機械学習手法のほうが許容範囲内の精度をより低コストで生成できる場合も多い。

ディープラーニングの良いところは学習には時間がかかるものの推論には時間がかからないところである。最先端の学習済みモデルを特徴量抽出器として用いることで深層学習の表現力を活かしつつ自然言語のような非線形で複雑なデータの分析を古典的機械学習手法で低コストで実現できることを示すことができたと思う。
