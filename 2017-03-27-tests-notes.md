---
title: "Statistics notes"
author: "Isabel María Villalba Jiménez"
date: 2017-03-27 17:31:00 -0700
layout: page
output:
  md_document:
    variant: markdown_github
    preserve_yaml: TRUE
    md_extensions: +emoji+citations+header_attributes
link-citations: yes
#https://www.rdocumentation.org/packages/rmarkdown/versions/1.3/topics/md_document
#My header {-} = # My header {.unnumbered}

navigation_weight: 2
---
# z-test, t-test, ANOVA and chi-squared tests
-----------
## Variance

### Descriptive

-   **SD**: Standard Deviation, *σ*²
-   $s^2\_n=\\frac{1}{n}\\sum(x\_i - \\bar{x})^2$
-   **Bessel's correction**
    -   $s^2=s^2\_n\\frac{n}{n-1}=\\frac{1}{n-1}\\sum(x\_i - \\bar{x})^2$

### Inferential

-   **SE**: Standard Error
-   $SE\_{\\bar{x}} = \\frac{s}{\\sqrt{n}}$
-   *s*<sup>2</sup> is an estimation of *σ*<sup>2</sup> the variance of the population.
-   The higher the number of elements of the sample, the lower the SE.

### Margin of error

$$ME = \\pm z^\* \\cdot SE = \\pm z^\* \\cdot\\frac{\\sigma}{\\sqrt{n}}$$

Types of tests
==============

<table style="width:100%;">
<colgroup>
<col width="14%" />
<col width="23%" />
<col width="18%" />
<col width="32%" />
<col width="10%" />
</colgroup>
<thead>
<tr class="header">
<th align="left"></th>
<th align="center">degrees of freedom</th>
<th align="left">Obvjective</th>
<th align="left">Conditions</th>
<th align="center">Formula</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left"><em>One sample</em></td>
<td align="center"></td>
<td align="left"></td>
<td align="left"></td>
<td align="center"></td>
</tr>
<tr class="even">
<td align="left">z-test</td>
<td align="center">-</td>
<td align="left"><span class="math inline">$\bar{x}$</span> vs <span class="math inline"><em>μ</em></span></td>
<td align="left">- Normal distribution <br> -<span class="math inline"><em>σ</em></span> and <span class="math inline"><em>μ</em></span> are known</td>
<td align="center"><span class="math inline">$z=\frac{\bar{x}-\mu}{\frac{\sigma}{\sqrt{n}}}$</span></td>
</tr>
<tr class="odd">
<td align="left">t-test</td>
<td align="center">n-1</td>
<td align="left"><span class="math inline">$\bar{x}$</span> vs <span class="math inline"><em>μ</em></span></td>
<td align="left">- Normal distribution <br> -<span class="math inline"><em>σ</em></span> unknown <br> -<span class="math inline"><em>μ</em></span> known</td>
<td align="center"><span class="math inline">$t=\frac{\bar{x}-\mu}{\frac{s}{\sqrt{n}}}$</span></td>
</tr>
<tr class="even">
<td align="left"><em>Two sample</em></td>
<td align="center"></td>
<td align="left"></td>
<td align="left"></td>
<td align="center"></td>
</tr>
<tr class="odd">
<td align="left">t-test <br> independent samples</td>
<td align="center"><span class="math inline"><em>d</em><em>f</em><sub>1</sub> + <em>d</em><em>f</em><sub>2</sub> = <em>n</em><sub>1</sub> + <em>n</em><sub>2</sub> − 2</span></td>
<td align="left">- <span class="math inline"><em>σ</em><sub>1</sub></span>,<span class="math inline"><em>σ</em><sub>2</sub></span> unknown <br> - <span class="math inline"><em>σ</em><sub>1</sub> ∼ <em>σ</em><sub>2</sub></span></td>
<td align="left">- Normal distribution <br> -<span class="math inline">$s^2_p=\frac{(n_1-1)s^2_1+(n_2-1)s^2_2}{n_1+n_2-2}$</span></td>
<td align="center"><span class="math inline">$t=\frac{\bar{x}_1-\bar{x}_2}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}$</span></td>
</tr>
<tr class="even">
<td align="left">t-test <br> dependent samples</td>
<td align="center"><span class="math inline"><em>n</em> − 1</span></td>
<td align="left">- <span class="math inline"><em>σ</em><sub>1</sub></span>,<span class="math inline"><em>σ</em><sub>2</sub></span> unknown <br> - <span class="math inline"><em>σ</em><sub>1</sub> ∼ <em>σ</em><sub>2</sub></span> <br> <span class="math inline"><em>d</em><sub><em>i</em></sub> = <em>x</em><sub>1<em>i</em></sub> − <em>x</em><sub>2<em>i</em></sub></span></td>
<td align="left">- Normal distribution <br> - 2 dependent samples (pre-treatment, post-treatment) <br> <span class="math inline">$s_d= \sqrt{\frac{\sum(d_i-\bar{d})^2}{n}}$</span></td>
<td align="center"><span class="math inline">$t=\frac{\bar{d}}{\frac{s_d}{\sqrt{n}}}$</span></td>
</tr>
<tr class="odd">
<td align="left"><em>Three or more samples</em></td>
<td align="center"></td>
<td align="left"></td>
<td align="left"></td>
<td align="center"></td>
</tr>
<tr class="even">
<td align="left">One way <br> ANOVA</td>
<td align="center"><span class="math inline"><em>d</em><em>f</em><sub><em>b</em><em>t</em><em>w</em></sub> = <em>k</em> − 1</span> <br> <span class="math inline"><em>d</em><em>f</em><sub><em>w</em></sub> = <em>N</em> − <em>K</em></span> <br> <br> <span class="math inline"><em>N</em> = ∑<em>n</em><sub><em>k</em></sub></span> <br> N = number of elements <br> k= number of groups</td>
<td align="left">- diff 3 or more population means</td>
<td align="left">- Normal distribution <br> -<span class="math inline"><em>s</em><sub>1</sub><sup>2</sup></span>, <span class="math inline"><em>s</em><sub>2</sub><sup>2</sup></span> sample variances</td>
<td align="center"><span class="math inline">$F= \frac{\frac{SS_{btw}}{df_{btw}}}{\frac{SS_w}{df_w}} = \frac{\sum n_k(\bar{x}_k-\bar{x}_G)^2/(K-1)}{\sum_{k=1}^{K}\sum_{i=1}^{n_k}(x_i-\bar{x}_k)^2/(N-K)}$</span></td>
</tr>
</tbody>
</table>

ANOVA
=====

$$MS\_{between}=\\frac{SS\_{btw}}{df\_{btw}}= \\frac{\\sum\_{k=1}^{K}n\_k(\\bar{x}\_k-\\bar{x}\_G)^2}{K-1}$$

$$MS\_{within}= \\frac{SS\_w}{df\_w} = \\frac{\\sum\_{k=1}^{K}\\sum\_{i=1}^{n\_k}(x\_i-\\bar{x}\_k)^2}{N-K}$$

for N the total number of elements and K the total number of groups.

$$F= \\frac{\\frac{SS\_{btw}}{df\_{btw}}}{\\frac{SS\_w}{df\_w}} = \\frac{\\sum n\_k(\\bar{x}\_k-\\bar{x}\_G)^2/(K-1)}{\\sum\_{k=1}^{K}\\sum\_{i=1}^{n\_k}(x\_i-\\bar{x}\_k)^2/(N-K)}$$

### F-statistic characteristics

-   ∀ &gt; 0
-   + skewed

<p align="center">
<img src="https://saylordotorg.github.io/text_introductory-statistics/section_15/75fd6a2d869b9403de8408b42a054db9.jpg" alt="F-distribution" width="300">
</p>
Cohen's d for multiple comparisons -&gt; effect size
----------------------------------------------------

$$d=\\frac{\\bar{x}\_1-\\bar{x}\_2}{\\sqrt{MS\_{within}}}=\\frac{\\bar{x}\_1-\\bar{x}\_2}{\\sqrt{\\frac{\\sum(x\_i-\\bar{x}\_k)^2}{N-k}}}$$

Eta-squared *η*<sup>2</sup>
---------------------------

*η*<sup>2</sup> : proportion of total variation due to between group differences (explained variation)

$$\\eta^2=\\frac{SS\_{between}}{SS\_{total}}$$

Correlation coefficient (Pearson's r)
=====================================

### For a population

[Reference:Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
$$ \\rho\_{X,Y}= \\frac{cov(X,Y)}{\\sigma\_X\\cdot \\sigma\_Y}$$
 Pearson's correlation coefficient when applied to a population is commonly represented by the Greek letter ρ () and may be referred to as the populatiPearson's correlation coefficient when applied to a population is commonly represented by the Greek letter *ρ* and may be referred to as the population correlation coefficient or the population Pearson correlation coefficient.

Where
*c**o**v*(*X*, *Y*)=*E*\[(*X* − *μ*<sub>*X*</sub>)(*Y* − *μ*<sub>*Y*</sub>)\]
 for *E*\[*X*\] the expected value of X or mean of X.

When |*ρ*|=1 this means that data lies in perfect line.

### For a sample

Pearson's correlation coefficient is represented as *r* when applied to a sample and it is called *sample correlation coefficient* or *sample Pearson correlation coefficient*.

$$r= \\frac{\\sum^n\_{i=1}(x\_i-\\bar{x})(y\_i - \\bar{y})}{\\sqrt{\\sum^n\_{i=1}(x\_i-\\bar{x})^2}\\sqrt{\\sum^n\_{i=1}(y\_i-\\bar{y})^2}}$$

Hypothesis testing
------------------

$$\\begin{align\*}H\_o:&\\rho=0 \\\\ H\_A: &\\rho&lt;0 \\\\ &\\rho &gt;0 \\\\ &\\rho\\neq0 \\end{align\*}$$

### Student's t-distribution

$$t=r\\sqrt{\\frac{n-2}{1-r^2}}$$

Which is a `t-distribution` with *d**f* = *n* − 2, for n the number of elements in the sample.

Regression
==========

Linear regression: $\\hat{y}=a+bx$

$$b = \\frac{\\sum\_{i=1}^n(x\_i-\\bar{x})(y\_i-\\bar{y})}{\\sum\_{i=1}^n(x\_i-\\bar{x})^2} = r \\frac{s\_y}{s\_x}$$

**SD**: how far values will fall from the regression line.

*SD of the estimate* = $\\sqrt{\\frac{\\sum(y-\\hat{y})^2}{n-2}}$

$residuals=\\sum{(y\_i -\\hat{y\_i})^2}$

Line of best fit: minimizes residuals

<p align="center">
<img src="http://maths.nayland.school.nz/Year_13_Maths/3.9_Bivariate_data/Scatterplot_images/10_Reg5.gif" alt="F-distribution" width="200" height="200">
</p>
Confidence interval (CI)
------------------------

### Population

-   *β*<sub>0</sub>: population *y*<sub>*i**n**t*</sub>
-   *β*<sub>1</sub>: population slope

### Sample

-   a : sample *y*<sub>*i**n**t*</sub>
-   b : sample slope

$$\\begin{align\*}H\_o:&\\beta\_1=0 \\\\ H\_A: &\\beta\_1\\neq0 \\\\ &\\beta\_1&gt;0 \\\\ &\\beta\_1&lt;0 \\end{align\*}$$

*d**f* = *n* − 2, for n the number of elements in the sample.

*χ*<sup>2</sup> test for independence
=====================================

$$\\chi^2 = \\sum\_{i=1}^n{\\frac{(f\_{oi} - f\_{ei})^2}{f\_{ei}}}$$

### 1 variable with ≠ responses

*d**f* = *n* − 1

### 2 or more variables

*d**f* = (*n*<sub>*r**o**w**s*</sub> − 1)(*n*<sub>*c**o**l**s*</sub> − 1) for N the number of categories

$$f\_{ei} = \\frac{(column\\; total)(row\\,total)}{grand \\,total}$$

-   ∀ *χ*<sup>2</sup> &gt; 0
-   ∀ one-directional test
-   *χ*<sub>*c**r**i**t*</sub><sup>2</sup>
