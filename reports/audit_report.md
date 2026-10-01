# Auditing Income Prediction Models for Gender Bias in Employment Decisions

I audited a logistic regression income classifier to examine whether strong aggregate prediction performance can coexist with unequal outcomes across recorded sex groups. The model achieves 85.49% test accuracy and 0.9093 ROC AUC, but its positive prediction rate is 8.53% for Female records and 26.02% for Male records. Similar positive predictive value across groups does not resolve these differences. The evidence supports further research and governance review; it does not validate employment screening.

## Research question and scope

The project examines demographic parity, equal opportunity, equalized odds, and predictive parity, then connects those metrics to global explanations, local explanations, and hypothetical recourse. Employment decisions motivate the discussion because automated scores can affect opportunity. The actual task is prediction of a historical income label. The data do not measure job qualification, hiring outcomes, or future employee performance.

The dataset calls its protected attribute `sex` and records Female and Male categories. I use those categories as recorded, without treating them as a comprehensive measure of gender identity. References to gender bias describe the motivating concern, while all numerical comparisons concern the recorded categories.

## Data and exploratory findings

I use the supplied 32,561-record Adult file with 14 predictors and one income label. The UCI release contains 48,842 records across its data files and was extracted from the 1994 Census database. The separate official test file is not part of this experiment. Instead, I split the included file into 26,048 training records and 6,513 test records.

Missing values are represented by `?`: 1,836 in workclass, 1,843 in occupation, and 583 in native country. I retain all records and replace those entries with the category Unknown. This avoids excluding incomplete records, but the data do not establish the causes of missingness. Treating missingness as a category can itself influence predictions and should be reviewed.

The exploratory plots show class imbalance and differences in observed income across recorded sex groups. Education and weekly work hours are also associated with income. These associations may reflect many social and economic processes; the plots do not establish a causal explanation. Marital status, relationship, occupation, and work hours are plausible proxies worth testing because they may retain group information even if the explicit sex attribute is removed.

## Model and evaluation design

I convert the income target to 1 for income above $50K and 0 otherwise, one-hot encode categorical variables, and create a stratified 80/20 split with seed 42. A StandardScaler is fitted on training records and applied to both splits. Logistic regression uses C of 1.0, the lbfgs solver, and a maximum of 5,000 iterations. The default prediction threshold is 0.50.

The baseline establishes category vocabulary before splitting to preserve the original experiment. This reveals test category presence, although model fitting and scaling use only training records. A future evaluation should fit the encoder entirely on training data. The survey-related fnlwgt field is retained as a predictor; it is not used to weight training or evaluation. Reported metrics describe these records and are not population estimates.

| Overall test metric | Value |
| --- | ---: |
| Accuracy | 0.8549 |
| Precision | 0.7365 |
| Recall | 0.6186 |
| F1 | 0.6724 |
| ROC AUC | 0.9093 |

Recall of 0.6186 indicates that aggregate accuracy does not capture all missed high-income cases. The repository saves the majority-class baseline alongside these measures to put accuracy in context.

## Fairness findings

| Measure | Female | Male | Absolute gap |
| --- | ---: | ---: | ---: |
| Test records | 2,158 | 4,355 | — |
| Actual positive rate | 0.1135 | 0.3038 | 0.1903 |
| Positive prediction rate | 0.0853 | 0.2602 | 0.1749 |
| True positive rate | 0.5673 | 0.6281 | 0.0608 |
| False positive rate | 0.0235 | 0.0996 | 0.0761 |
| Positive predictive value | 0.7554 | 0.7335 | 0.0220 |

Demographic parity compares positive prediction rates. The Female-to-Male positive prediction rate ratio is approximately 0.3277. The observed base rates differ too, so the prediction disparity cannot be interpreted independently of the target and historical data. A base-rate difference does not determine whether repurposing this target for another decision would be appropriate.

Equal opportunity compares true positive rates. Female records have a lower true positive rate and a higher false negative rate: approximately 0.4327 versus 0.3719. In this dataset, that means high-income Female records are missed more often. It does not mean qualified applicants are missed, because qualification is not measured.

Equalized odds examines both true positive and false positive rates. The model also has a higher false positive rate for Male records. Predictive parity appears closer: precision differs by only 0.0220. These metrics concern different aspects of model behavior, so a small precision gap cannot stand in for equal selection rates or equal error rates.

## Threshold sensitivity

I evaluate thresholds from 0.10 to 0.90 in steps of 0.05 on the held-out test records. Within this grid, accuracy is highest at 0.50. The demographic parity gap falls from 0.4145 at 0.10 to 0.0273 at 0.90. Increasing the threshold also reduces access to positive predictions and changes recall and error rates. A smaller absolute gap can reflect fewer positive predictions rather than a broadly useful decision rule.

This experiment illustrates sensitivity, rather than selecting a deployable threshold. Using its results to choose a policy would require a separate validation set, a stated decision objective, uncertainty estimates, and independent test evaluation. If a future employment task has a valid qualification label, equal opportunity could be a relevant objective for avoiding missed qualified people. That preference remains a policy choice tied to the task and its harms.

## SHAP and LIME explanations

The leading individual SHAP features are marital status Married civ spouse, marital status Never married, capital gain, age, and education number. Aggregating absolute contributions by original field makes marital status, occupation, education, and relationship prominent. Explicit sex features also contribute. The results suggest areas for ablation and proxy analysis; they do not establish a causal pathway or prove that removing a feature will resolve disparity.

SHAP values for this linear classifier are in log-odds units. They depend on the background distribution, preprocessing, and correlated features. Summing absolute contributions within an original field can advantage fields with many one-hot categories. The grouped ranking should be interpreted with that aggregation choice in mind.

I use LIME for one positive prediction, with probability approximately 0.9941, and one negative prediction just below the decision boundary, with probability approximately 0.4996. The latter has an observed positive label and is therefore a false negative. Capital gain is an influential factor in the local explanations. Several rare country and workclass indicators also appear, which can be difficult to translate into a useful explanation for a person.

LIME approximates behavior near a selected input using sampled perturbations. The rerun sets seed 42 for repeatability, but the explanation is not causal and can vary with settings. Independent perturbations of one-hot columns can produce combinations that do not represent valid records. Two examples cannot establish explanation quality across the population.

## Hypothetical recourse

The recourse example is a 51-year-old Male record with Private workclass, Assoc voc education, Craft repair occupation, and 40 work hours per week. The model assigns a probability of approximately 0.4996 despite an observed high-income label. I enumerate increases in education or work hours, changes in occupation or workclass, and selected combinations. Immutable personal attributes are held fixed; education label and numeric level are changed together.

Increasing weekly hours from 40 to 45 changes the model probability to approximately 0.5371. Changing occupation to Exec managerial changes it to approximately 0.6628. Both flip the predicted class. These experiments test the model's response to hypothetical inputs. They do not establish that the changes are feasible, socially available, or causally effective. Changing occupation may conflict with other fields, and working longer may be limited by health, caregiving, or job availability.

The example was deliberately selected near the boundary, so it may be easier to flip than other negative predictions. A fuller study would assess many records, enforce valid combinations, define costs and constraints, and compare feasibility across groups. People should not be blamed for failing to carry out an unrealistic model-generated change.

## Regulatory context

Regulatory applicability depends on actual deployment, jurisdiction, roles, and effective provisions. This benchmark is not a deployed employment system, and the audit does not determine legal compliance. The following connections identify questions that an organization would need to examine for a separate employment application.

The U.S. Uniform Guidelines use the four-fifths ratio as an adverse-impact screening rule of thumb. EEOC guidance explains that it is not a legal definition or a final determination of unlawful discrimination. The benchmark ratio of 0.3277 would be a signal to investigate if comparable predictions became selection decisions, but it is not an observed hiring impact ratio. Job-related validity and the full selection process would matter.

GDPR Article 22 addresses certain decisions based solely on automated processing with legal or similarly significant effects, subject to specified exceptions and safeguards. Other provisions address transparency and impact assessment. An actual application would need to assess territorial scope, legal basis, whether the decision is covered, and the relevant obligations. A SHAP plot by itself does not establish meaningful transparency or a usable review process.

The EU AI Act identifies specified recruitment and worker-management uses in Annex III, subject to its classification conditions. Its data governance, transparency, and human oversight requirements provide a relevant framework for examining an employment application. The current implementation timeline should be checked before use; an income benchmark's existence alone does not make it a covered high-risk deployment. The applicability of fundamental rights impact assessment obligations depends on the deployer and use.

NYC Local Law 144 sets requirements for covered automated employment decision tools, including a bias audit, publication of results, and notices. This project is an internal benchmark audit and does not establish that those requirements have been met. Scope, independence, data, and the complete audit process would need separate review.

## Governance recommendations

I recommend keeping the present model within benchmark research and fairness testing. An employment application would first need a target that measures the intended decision, data that represent the relevant population, and evidence that the model is suitable for that use. High accuracy on Adult cannot provide that evidence.

Before considering a separate application, an organization should document intended use and affected groups, review sensitive and proxy variables, compare models with and without those variables, and examine several fairness metrics. Independent validation should quantify uncertainty, calibration, subgroup error rates, and intersectional outcomes. The organization should establish responsibilities for monitoring, review, and stopping use when approved conditions are no longer met.

Human review should be meaningful, with access to relevant information and authority to reconsider outcomes. People should be able to understand the decision process, correct inaccurate data, request review, and contest decisions where applicable. Explanations should describe model associations clearly and avoid presenting hypothetical recourse as guaranteed personal advice. Regulatory review and any required independent audit are separate from this project's numerical evaluation.

## Example communication for a future system

If a separately validated system were introduced, a plain-language notice should identify its purpose, data inputs, role in the decision, known limitations, and the route for requesting review. An example statement is: “The system produces a prediction from recorded information. A prediction may be inaccurate and does not provide a complete assessment of your abilities. You can ask us to explain its role, review the information used, and reconsider the decision through the stated review process.” The wording and actual safeguards would need to match the real system and applicable requirements.

## Limitations and further work

The findings concern one historical dataset, one classifier, and one random split. No confidence intervals, repeated-split results, external validation, mitigation comparisons, or intersectional audit are included. Recorded categories omit many relevant identities and conditions. Preprocessing vocabulary uses the full included file; metrics are unweighted; explanations depend on correlated features; recourse lacks causal and feasibility constraints. The threshold sweep and local examples are descriptive analyses of the test set.

Next steps are to fit all preprocessing on training data, preserve a separate validation set, test ablations and fairness interventions, estimate uncertainty, inspect calibration, and evaluate independent data with a valid target for the intended use. A broader recourse evaluation should document costs, enforce consistent categories, and include affected people's constraints. These are proposed extensions, not completed experiments.

## Reproducibility

The repository contains the unchanged data bytes, a notebook, a matching runnable Python analysis, saved tables and figures, and dependency specifications. Run python src/audit.py from the repository root after installing requirements.txt. The results directory stores overall metrics, subgroup metrics and confusion counts, disparity tables, threshold sensitivity, SHAP importance, LIME explanations, recourse candidates, and run metadata. Seeds, input checksum, split sizes, and package versions are recorded in run_metadata.json.

## References

- Becker, B. and Kohavi, R. (1996). Adult dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. https://arxiv.org/abs/1705.07874
- Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). Why Should I Trust You. https://arxiv.org/abs/1602.04938
- Pedregosa et al. (2011). Scikit-learn Machine Learning in Python. https://www.jmlr.org/papers/v12/pedregosa11a.html
- EEOC. Questions and Answers on the Uniform Guidelines. https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines
- GDPR Regulation EU 2016/679. https://eur-lex.europa.eu/eli/reg/2016/679/oj
- EU AI Act Regulation EU 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Commission. AI Act framework and implementation timeline. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- NYC Department of Consumer and Worker Protection. Automated Employment Decision Tools. https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page
