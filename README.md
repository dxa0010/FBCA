# Fixed Budget Context Aware Active Learning

## Setting

+ OS: Ubuntu 18.04
+ CPU: Core i7
+ GPU: GTX 1080 Ti
+ using Nvidia-docker

``` sh

pip install -r requirements.txt

```

Datasets are listed in references in the paper．

## how to use

### If you only want to check the result

<!-- "compare_input_domain.ipynb"で 学習ドメイン（人工データ）を変化させた時の結果の違いを確認できる．
"compare_methods.ipynb"で 手法によるの結果の違いを確認できる．
またoracleとの比較では，"result/ral_compare_oracle..."が提案法の実験結果，"result/oracle_random..."がrandomの結果，"oracle_trip..."がoracleの結果として出力しているので，そちらで確認できる -->

In "compare_input_domain.ipynb", you can see the difference in the result when the learning domain (artificial data) is changed.
In "compare_methods.ipynb", you can see the difference of the result by the method.
Also, in comparison with oracle, "result / ral_compare_oracle ..." is the result of the proposed method, "result / oracle_random ..." is the result of random, and "oracle_trip ..." is the result of oracle. So you can check there.

### If you want to do an experiment

In order to carry out a follow-up experiment, an overview of each file is given.

+ artificial_data_maker.py

Generate artificial data.
The data for learning and the data for testing are generated here.
First of all.

+ learning.py

This is the code for model learning.

+ artificial_data_test, real_world_data_test, oracle_test

<!-- それぞれ，人工データでの実験，実データでの実験，Oracleとの比較の実験を行うコードである．
結果はresultに出力される． -->

These are codes for performing experiments with artificial data, experiments with real data, and experiments comparing with Oracle, respectively.
The result is output to "result/".

+ compare

The code of the comparison method is arranged.
<!-- "integrate.py" ですべての比較手法の結果を出力できる． -->
"integrate.py" can output the results of all comparison methods.