# Project Structure
DataMining_GroupProject_<GroupMembersName>/  (This is the root of your repository) 
├── data/ 
│   ├── raw/ 
│   ├── transformed/ 
│   └── final/ 
├── notebooks/ 
│   ├── 1_extract_transform.ipynb [cite: 18]
│   ├── 2_exploratory_analysis.ipynb [cite: 19]
│   ├── 3_data_mining.ipynb [cite: 20]
│   └── 4_insights_dashboard.ipynb [cite: 21]
├── report/ 
│   └── executive_summary.pdf [cite: 24]
├── requirements.txt 
├── .gitignore 
└── README.md [cite: 27]
```markdown
# End-to-End Data Mining Project Plan

This plan outlines a detailed approach to successfully complete your DSA 2040A Group Project, ensuring all requirements are met and contributions are clearly documented for full marks.

## Overall Strategy for Full Marks

- **Strict Adherence to Requirements**: Go through the entire project document and create a checklist of every single requirement (folder structure, file names, content, collaboration rules, README contents). Tick them off as you complete them.

- **Proactive Collaboration & Communication**:
  - Establish clear communication channels (e.g., WhatsApp group, Discord).
  - Schedule regular (e.g., twice a week) short stand-up meetings to discuss progress, roadblocks, and next steps.
  - Assign specific tasks with deadlines to each member.

- **Version Control Best Practices (GitHub is Key!)**:
  - *Commit Early, Commit Often*: Encourage small, frequent commits with clear, descriptive messages. This makes tracking contributions easier.
  - *Branching (Optional but Recommended)*: For larger features or individual work, consider creating separate branches and merging them into main after review. This prevents conflicts.
  - *Pull Requests (Optional but Recommended)*: Use PRs for code review before merging, ensuring quality and giving everyone visibility.

- **Documentation is Paramount**:
  - `README.md`: Keep it updated constantly as the project progresses. This is your project's front page.
  - *Notebook Comments*: Use markdown cells and code comments extensively to explain your logic, findings, and especially to mark individual contributions.
  - *Executive Summary*: Start outlining this early, even with bullet points, and refine it in Week 5.

- **Quality over Quantity (for Data Mining Techniques)**: Focus on thoroughly implementing and evaluating 2-3 chosen techniques rather than superficially covering many.

- **Actionable Insights**: The final dashboard and executive summary should clearly present actionable insights that inform decision-making, not just raw data.

## 5-Week Roadmap & Detailed Plan

### Week 1: Kickoff & Dataset Selection

**Goal**: Project setup, team roles, dataset finalization, and initial ETL pipeline plan.

**Tasks**:

1. **Team Formation & Roles**:
   - Confirm group members.
   - Appoint one member to create the GitHub repository.
   - Assign roles: ETL Lead, Analyst, Visualizer, Documenter. Ensure roles are distributed fairly and align with individual strengths.

2. **GitHub Repository Setup**:
   - Create the public GitHub repository with the exact name format: `DSA2040A_DataMining_<GroupName>`.
   - Add all other group members as collaborators with write access.
   - Clone the repo to local machines.
   - Create the required folder structure: `data/raw/`, `data/transformed/`, `data/final/`, `notebooks/`, `report/`.
   - Add initial placeholder files: `requirements.txt`, `.gitignore`, `README.md`, and empty `.ipynb` files for each notebook (`1_extract_transform.ipynb`, etc.).

3. **Dataset Selection**:
   - Brainstorm and choose a relevant dataset (e.g., e-commerce, health, finance, education) or decide to expand on the previously cleaned midterm data.
   - Clearly define the project goal and the type of insights you aim to uncover.

4. **ETL Pipeline Finalization/Expansion**:
   - Review the midterm ETL pipeline.
   - Identify areas for expansion or improvement based on the chosen dataset and project goals.
   - Outline the steps for initial data extraction and transformation.

**Deliverables (Submit: project proposal in README.md)**:

- **GitHub Repo**: Public, correct naming, all members added.
- **README.md**:
  - Group name, member names (first name only), last 3 digits of ID.
  - Project summary (topic, questions asked).
  - Initial ETL summary (what data you'll use, initial cleaning steps).
  - Initial tools used (Pandas, etc.).
  - Instructions to run notebooks (even if empty for now).
  - *Crucially*: Add a "Team Members & Contributions" section with initial role assignments and a commitment to documenting individual contributions.
- **Contribution Tracking**:
  - Each member makes an initial commit (e.g., adding their name to README, creating a placeholder file) to demonstrate cloning and commit access.

### Week 2: Data Cleaning & Enrichment

**Goal**: Produce a clean, enriched dataset ready for analysis.

**Tasks**:

1. **Data Extraction & Initial Transformation**:
   - Implement the initial ETL steps in `1_extract_transform.ipynb`.
   - Load raw data.
   - Perform basic cleaning: handle missing values, correct data types, remove duplicates.

2. **Data Enrichment**:
   - Add calculated fields (e.g., `profit_margin`, `purchase_frequency`, `customer_lifetime_value`, `order_value_per_item`).
   - Consider feature engineering relevant to your chosen data mining techniques.

3. **Advanced Cleaning**:
   - Handle outliers (e.g., capping, removal, transformation).
   - Standardize formats (dates, text).
   - Remove noise/irrelevant data.

4. **Documentation**:
   - Thoroughly comment `1_extract_transform.ipynb` explaining each step.
   - *Individual Contribution*: Use markdown cells or comments to clearly mark sections worked on by specific team members (e.g., `# Section by [Member Name]: Data Type Conversion`).

**Deliverables (Submit: updated 1_extract_transform.ipynb)**:

- `1_extract_transform.ipynb`: Clean, well-commented, and produces a transformed dataset saved in `data/transformed/`.
- **Contribution Tracking**:
  - Each member makes at least one commit related to data cleaning/enrichment.
  - Commit messages should be clear (e.g., "Grace: Cleaned nulls in sales data," "Brian: Added profit_margin calculation").

### Week 3: Exploratory & Statistical Analysis (EDA)

**Goal**: Understand data patterns, relationships, and prepare for data mining.

**Tasks**:

1. **Visual EDA**:
   - Use Pandas, Seaborn, Matplotlib (and potentially Plotly for interactive plots) to create visualizations.
   - Explore distributions of key variables (histograms, box plots).
   - Visualize relationships between variables (scatter plots, pair plots).
   - Identify trends and patterns.

2. **Statistical Analysis**:
   - Calculate correlations between relevant features.
   - Perform group comparisons (e.g., t-tests, ANOVA if applicable).
   - Identify potential features for data mining models.

3. **Documentation**:
   - Interpret all plots and statistical results in markdown cells within `2_exploratory_analysis.ipynb`.
   - *Individual Contribution*: Clearly mark sections (e.g., `# Section by [Member Name]: Customer Demographics Analysis`).

**Deliverables (Submit: 2_exploratory_analysis.ipynb)**:

- `2_exploratory_analysis.ipynb`: Rich with visualizations, statistical analysis, and clear interpretations.
- **Contribution Tracking**:
  - Each member makes at least one commit related to EDA.
  - Commit messages reflect the specific EDA tasks performed.

### Week 4: Data Mining

**Goal**: Implement and evaluate chosen data mining techniques.

**Tasks**:

1. **Technique Selection (2-3)**:
   - Based on EDA and project goals, choose 2-3 appropriate mining techniques (e.g., Clustering: k-means, DBSCAN; Classification: Decision Trees, Logistic Regression; Association Rules; Time Series).
   - *Tip*: If your data is suitable for both clustering and classification, pick one of each for variety.

2. **Model Implementation**:
   - Implement the chosen techniques in `3_data_mining.ipynb`.
   - Prepare data for modeling (e.g., feature scaling, one-hot encoding).
   - Train your models.

3. **Model Evaluation**:
   - Evaluate results using appropriate metrics (e.g., for clustering: silhouette score; for classification: accuracy, precision, recall, F1-score, confusion matrix; for association rules: support, confidence, lift).
   - Discuss the strengths and weaknesses of each model.

4. **Documentation**:
   - Explain the rationale for choosing each technique.
   - Interpret model results thoroughly in markdown cells.
   - *Individual Contribution*: Clearly mark sections (e.g., `# Section by [Member Name]: K-Means Clustering Implementation`).

**Deliverables (Submit: 3_data_mining.ipynb)**:

- `3_data_mining.ipynb`: Contains implemented models, evaluation metrics, and clear interpretations.
- **Contribution Tracking**:
  - Each member makes at least one commit related to data mining.
  - Commit messages are specific to the models/evaluations performed.

### Week 5: Insight & Storytelling

**Goal**: Present actionable insights through a dashboard, executive summary, and presentation.

**Tasks**:

1. **Dashboard Creation**:
   - Build a mini-dashboard using Power BI, Jupyter Dash, Seaborn/Plotly, or even just well-structured plots in a notebook.
   - Focus on visualizing the most impactful insights derived from your EDA and data mining.
   - Ensure the dashboard is clear, concise, and visually appealing.

2. **Actionable Insights**:
   - Identify 3-5 actionable insights that directly answer your project questions and can inform decision-making for a hypothetical business/stakeholder.
   - Phrase these insights clearly and concisely.

3. **Executive Summary**:
   - Write a 1-2 page PDF executive summary.
   - Include: Project goal, key findings (your 3-5 actionable insights), brief methodology, and recommendations.
   - Ensure it's high-level and targets a non-technical audience.

4. **Final Presentation Deck**:
   - Prepare a presentation (PPTX/PBIX) summarizing your project.
   - Include key visuals from your notebooks and dashboard.
   - Practice the presentation as a group.

5. **Final Notebook Refinement**:
   - Ensure `4_insights_dashboard.ipynb` is complete, well-commented, and presents your dashboard and insights clearly.

6. **README.md Finalization**:
   - Update the README with final summaries of ETL, techniques used, and tools.
   - *Crucially*: Ensure the "Team Members & Contributions" section is fully populated, detailing who did what for each notebook/major section, and confirming that each member has made 3+ commits.

**Deliverables (Submit: 4_insights_dashboard.ipynb, executive_summary.pdf, presentation.pptx)**:

- `4_insights_dashboard.ipynb`: Contains the mini-dashboard and clear presentation of actionable insights.
- `executive_summary.pdf`: 1-2 pages, concise, high-level, actionable.
- `presentation.pptx`: Final pitch deck.
- Updated `README.md`: Comprehensive and accurate.
- **Contribution Tracking**:
  - Each member makes final commits for their contributions to the dashboard, executive summary, or presentation.
  - Verify all members have made at least 3 commits and their contributions are visible in the commit history and notebook comments.

## GitHub Collaboration & Contribution Scoring (Crucial for Full Marks)

- **Repository Ownership**: Ensure one member owns the repo and all others are collaborators with write access.

- **Commit History**:
  - `git log --pretty=format:"%an %s"` will be used by the grader. Make sure your commit messages are clear and your names are associated with your commits.
  - Aim for more than 3 commits per person. One commit per notebook or major section is a good guideline.

- **README.md**:
  - *DO NOT FORGET*: Add all names (first name only) and the last 3 digits of your ID.
  - The "Team Members & Contributions" section is vital. Explicitly state who led/contributed to which notebooks or major project phases (e.g., "Grace M. (Notebooks 2, 4) - Responsible for EDA & dashboard visualizations").

- **Individual Comments in Notebooks**:
  - This is a mandatory requirement to avoid a -5 mark penalty.
  - Use markdown cells or code comments like:
    ```
    # Section by Grace: Data Cleaning for Missing Values
    # ... (Grace's code) ...

    # Section by Brian: Feature Engineering - Profit Margin Calculation
    # ... (Brian's code) ...
    ```

- **Avoid "One Person Committed Everything"**: This results in a 20% group penalty. Ensure everyone is actively committing from their own GitHub account.

- **Final Check**: Before submission, ensure:
  - All notebooks are well-commented with individual contributions marked.
  - The README.md is comprehensive and up-to-date.
  - The GitHub repository is public and accessible to the grader.