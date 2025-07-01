# Credit-Risk-Model-Week5

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord emphasizes the importance of robust risk measurement and management frameworks for financial institutions. It introduces capital requirements based on the internal ratings-based (IRB) approach, which relies heavily on model-based estimations of credit risk components such as Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

This regulatory pressure necessitates transparent, interpretable, and well-documented models. Models used for regulatory capital calculation must be explainable to both internal stakeholders and external regulators. Simple models like logistic regression with Weight of Evidence (WoE) transformations are favored in regulated contexts because they allow auditors and regulators to trace back predictions to specific, understandable factors, ensuring compliance and accountability.

### Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world datasets, especially in microcredit or alternative credit scoring environments, there is no direct or reliable label for "default." In such cases, a proxy variable (e.g., missing payments after 60 days) is used to indicate likely default.

Creating this proxy is essential for supervised learning, but it introduces several business risks:

**Misclassification Risk**: A proxy may not truly represent a default, leading to biased predictions and unfair lending decisions.
**Regulatory Risk**: If models are based on loosely defined or opaque proxy variables, they may fail regulatory scrutiny.
**Operational Risk**: Actions taken based on flawed predictions (e.g., loan denial or pricing) can damage customer relationships and lead to lost revenue.

Thus, choosing and validating proxy variables must be done carefully, with domain expertise and data transparency.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In the context of regulated financial services, a key trade-off exists between:

**Simple, interpretable models** (e.g., logistic regression with WoE):

- Easy to explain and audit
- Compliant with regulatory requirements (Basel II/III, IFRS 9)
- Transparent decisions for customers
- May underperform on complex, non-linear data

**Complex, high-performance models** (e.g., Gradient Boosting, Random Forest, Deep Learning):

- Higher predictive power
- Ability to capture non-linearities and interactions
- Often act as "black boxes"
- Require advanced model interpretability tools (e.g., LIME, SHAP) to justify decisions

While high-performance models offer better accuracy, their adoption in credit scoring is limited unless paired with robust governance frameworks and interpretability tools. Financial institutions often start with interpretable models and gradually incorporate advanced methods with post-hoc explanation techniques and proper documentation.
