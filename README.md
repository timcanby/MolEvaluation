# MolEvaluation
This is a project to evaluate generated molecules:

sample of outputs:
Distance(gen*test): 0.003099027450687202
Distance(gen*gen): 0.5
Distance(test*test): 0.5851409978308026
Silhouette Score(n=2)(gen*gen): 1.0
Silhouette Score(n=2)(gen*test): 0.009099692068578386
valid 0.8333333333333334
unique@100 0.8
SNN/Test 0.45220685228705404
Frag/Test 0.3952847075210474
Scaf/Test 0.6324555320336758
SNN/TestSF 0.4334906339645386
Frag/TestSF 0.3006277963620376
Scaf/TestSF 0.017365828833749708
IntDiv 0.6949467039108276
IntDiv2 0.4769297423177694
Filters 0.8
logP 1.3082120000000008
SA 0.4954241886103563
QED 0.17304732217043892
weight 94.50380000000004
Novelty 1.0

# References
Parts of the code were reference the following repositories:
 * [moses](https://github.com/molecularsets/moses#metrics)
 * [Silhouette Score](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c)

