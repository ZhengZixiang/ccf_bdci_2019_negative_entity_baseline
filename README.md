# ccf_bdci_2019_negative_entity_baseline
以ABSA的思路去做，用的是BERT-SPC模型，也就是将数据处理成文本（标题+正文）和实体作为句子对，将是否负面作为情感标签。

由于文本比较长，所以比较粗暴地取了实体前后总长为max_seq_length（这里用的是512）做个截断，效果不如直接截取前面512长的文本效果好。BERT-wwm-ext理论和线下是比Google的bert-base-uncased有提升的，但是线上效果反而变差。代码里polarities_dim=2，理论上也应该是2，但实际设为3再跑线上效果反而更好，玄学。

infer时再将一个个实体去判断情感，这里参考了判断子串的后处理，能带来0.01的提升。跑出来0.86左右，[GeneZC](https://github.com/GeneZC/BERTFinanceNeg)做法一样，但是能跑到0.92，可能是我截文本数据处理问题较大，就这样吧，存个baseline。

## Reference
- https://github.com/EliasCai/CCF-negative-entity
- https://github.com/rjk-git/CCF_Negative_Financial_Information_and_Subject_Judgment
- https://github.com/songyouwei/ABSA-PyTorch
