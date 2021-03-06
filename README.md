# Bert-for-IR

作為IR(Information Retrieval)的最後一個作業，當然要搬出NLP方面遠近馳名的Bert來實作這個task。然而單純使用類神經網路無法在IR的task上面取得好的performance，在這個資料集上Bert的acc大約是30%左右，而傳統的BM25卻是39%左右。原因是對於每一個Query而言我們的positive sample實在太少了，以這次的資料集而言每個Query大概只有五十個以內的positive sample，很難期待類神經網路能從這麼少的sample學到太generic的資訊。

然而，這並不代表Bert在IR是完全沒有用的，我們可以用ensemble的方式結合BM25以及Bert，提高performance。因此在這個作業我們事先準備好了每一個Query使用BM25找回的前一千篇document以及其BM25的score，將這一千篇document去除此Query的postive document後當作negative sample，從中random sample出少數的文件標記成negative sample，以這種方式去訓練Bert才能有效的讓他學到positive 和 negative sample的差異。

最後我們將BM25的分數以及Bert的分數ensemble後，可以將acc提高到45%左右。若進一步ensemble XLNet，則可達到48%左右的accuracy。而RoBerta則因為在pretrain時沒有加入NSP(Next Sentence Prediction)，所以用來區分一串input sequence中Query和Document的token type ids沒有作用，其performance並沒有很好。
