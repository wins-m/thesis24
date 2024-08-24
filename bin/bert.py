import pandas as pd
import bert_pytorch

src = 'cache/sentences.10.pkl'
df = pd.read_pickle(src)

all_sentences = '\n'.join(df.sentence.apply(lambda x: ' \t '.join(x)).to_list())
out = 'cache/corpus.all'
with open(out, 'w', encoding='utf-8') as f:
    f.write(all_sentences)
print(out)


df1 = df.query("20210101 <= tradingdate <= 20220630")
sentences_small = '\n'.join(df1.sentence.apply(lambda x: ' \t '.join(x)).to_list())
out = 'cache/corpus.small'
with open(out, 'w', encoding='utf-8') as f:
    f.write(sentences_small)
print(out)