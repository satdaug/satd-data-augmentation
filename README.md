Dataset of Deep Learning Approach for Identifying and Categorizing Self-Admitted Technical Debt

Abstract:
Self-Admitted Technical Debt (SATD) refers to circumstances where developers annotate code comments, issues, pull requests, or other textual artifacts to explain why the existing coded implementation is not optimal. Past research has often used development artifacts to isolate SATD: shared datasets were produced where independent researchers have labeled a certain textual artifact as an instance of SATD or not (i.e., a binary classification). In other limited cases, various labels were created to categorize SATD into different types (design debt, test debt, etc.). 

All the previous research on SATD suffers from two common problems: it is based on shared (and widely adopted) datasets that are greatly unbalanced in terms of labels. Second, the majority of past SATD research has either adopted a binary classification of SATD, and way less often, delved in its type classification.

In this study, we utilize a data augmentation strategy to address the problem of imbalanced classes. We also employ a two-step approach to identify and categorize SATD on a variety of datasets derived from different artifacts. Based on earlier research, a deep learning architecture called BiLSTM is utilized for the binary identification of SATD. The BERT architecture is then utilized to categorize different types of SATD. We provide the balanced classes as a contribution for future TD researchers, and we also show that the performance of SATD identification and categorization using deep learning and our two-step approach is competitive with (or better than) baseline deep learning approaches.

This dataset has undergone a data augmentation process using the AugGPT technique. Meanwhile, the original dataset can be downloaded via the following link: https://github.com/yikun-li/satd-different-sources-data
