# Atlas-based Semantic Segmentation of Prostate Zones - MICCAI 2022

Jiazhen Zhang<sup>1</sup>; Rajesh Venkataraman<sup>2</sup>, Ph.D.; Yihuan Lu<sup>6</sup>, Ph.D.; Lawrence H. Staib<sup>1,3,4</sup>, Ph.D.; John A. Onofrey<sup>1,2,4</sup>, Ph.D.

[Cite this article](#cite-this-article)
to do

## Abstract

Segmentation of the prostate into specific anatomical zones is important for radiological assessment of prostate cancer in magnetic resonance imaging (MRI).
Of particular interest is segmenting the prostate into two regions of interest: the central gland (CG) and peripheral zone (PZ).
In this paper, we propose to integrate an anatomical atlas of prostate zone shape into a deep learning semantic segmentation framework to segment the CG and PZ in T2-weighted MRI.
Our approach incorporates anatomical information in the form of a probabilistic prostate zone atlas and utilizes a dynamically controlled hyperparameter to combine the atlas with the semantic segmentation result.
In addition to providing significantly improved segmentation performance, this hyperparameter is capable of being dynamically adjusted during the inference stage to provide users with a mechanism to refine the segmentation. 
We validate our approach using an external test dataset and demonstrate Dice similarity coefficient values (mean $\pm $ SD) of 0.91 $\pm $ 0.05 for the CG and 0.77 $\pm $ 0.16 for the PZ that significantly improves upon the baseline segmentation results without the atlas.


![Example Segmentation Results](/resources/images/SegmentationExamples.png "CG and PZ segmentation results.")

## Author information

**Affiliations**

<sup>1</sup>	Department of Radiology and Biomedical Imaging, Yale University, New Haven, CT, USA 

<sup>2</sup>	Department of Urology, Yale University, New Haven, CT, USA

<sup>3</sup>	Department of Electrical Engineering, Yale University, New Haven, CT, USA

<sup>4</sup>	Department of Biomedical Engineering, Yale University, New Haven, CT, USA

<sup>5</sup>  Eigen Health, Grass Valley, CA, USA

## Cite this article
to do

### DOI
to do
