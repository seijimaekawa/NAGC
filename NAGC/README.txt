examle.ipynbを見ながら使い方を知ってもらうのが良いと思います.

また新しいデータを追加したい場合は, dataの中にあるような形式で保存をしてください.
注意点としては, 属性ファイル(.content)は以下のような並びです.
node1 \t att1 \t att2 \t ... \t attm \t true_clus \n
node2 \t att1 \t att2 \t ... \t attm \t true_clus \n
...

のような形です. 
正解データがない場合はtrue_clusにダミーを入れて
NMIやARIの評価の部分を実行しないでください.

kmeans初期化を行う場合
example.ipynbを動かす前に, data内に任意のデータを置いて
python init_kmeans.py -name DATANAME -k NUMBERofCLUSTERS
を実行してください. 
k1, k2の両方について実行してください.