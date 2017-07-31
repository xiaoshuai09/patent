"On Predictive Patent Valuation: Forecasting Patent Citations and Their Types"
############################################


Code accompanying the paper ["On Predictive Patent Valuation: Forecasting Patent Citations and Their Types"](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14385)

## Prerequisites

- Computer with Linux or OSX
- Language: python2.7

## Notes

- patent3 is the sample data: each patent has four line, the first row is self-citation, the second is other-citation, the three is the intenvation year of patents and the fourth are features, include "assignee_type", "n_inventor", "n_claim", "n_backward", "ratio_cite", "generality", "originality",etc.
- Raw dataset can be downloaded via: https://pan.baidu.com/s/1skNB6Qp

## Paper Abstract:
Patents are widely regarded as a proxy for inventive output which is valuable and can be commercialized by various means. Individual patent information such as technology field, classification, claims, application jurisdictions are increasingly available as released by different venues. This work has relied on a long-standing hypothesis that the citation received by a patent is a proxy for knowledge flows or impacts of the patent thus is directly related to patent value. This paper does not fall into the line of intensive existing work that test or apply this hypothesis, rather we aim to address the limitation of using so-far received citations for patent valuation. By devising a point process based patent citation type aware (self-citation and non-self-citation) prediction model which incorporates the various information of a patent, we open up the possibility for performing predictive patent valuation which can be especially useful for newly granted patents with emerging technology. Study on real-world data corroborates the efficacy of our approach. Our initiative may also have policy implications for technology markets, patent systems and all other stakeholders. The code and curated data will be available to the research community.
