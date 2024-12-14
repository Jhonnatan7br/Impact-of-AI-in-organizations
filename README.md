![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/d87a7990-b7c8-4769-a55e-9e028f2ca039) 
 
Official Document: https://1drv.ms/b/c/c91ef16cd60b809e/EZ6AC9Zs8R4ggMmNiwAAAAABa-Su2Cwlg6N9bias7ErzSw?e=x5fV2s 
 
# Evolution of technology perception and industries' adoption
 
How could machine learning and AI impact organizations?
This question is particularly pertinent in modern business management research, as it delves into a critical aspect of organizational dynamics in the face of advancing technologies. 

I've chosen to focus on the impact of machine learning and AI tools because they've already begun to reshape how work is organized within organizations, whether through the automation of routine tasks, augmentation of decision-making processes, or the emergence of entirely new roles and responsibilities centered around AI integration.
 
![image](https://github.com/user-attachments/assets/d10456d3-da6a-4a6f-88ed-80cd0d919f86)

![image](https://github.com/user-attachments/assets/4592e7f8-edd7-473d-a851-ab506600c4f7)

# Vader sentiment Analysis Heuristics

The sentiment analysis results were statistically analyzed using Scipy and Scikit-Learn libraries.
This involved calculating average sentiment scores across different topics and keyword clusters,
allowing for the identification of trends in how AI research is perceived across various
organizational contexts. (Scikit-learn: Machine Learning in Python, Pedregosa et al. 2011) The
statistical insights were supplemented with qualitative observations derived from the manual
extraction of key concepts, adding depth to the interpretation of the data.

![image](https://github.com/user-attachments/assets/1f396295-f87d-4d04-9ff5-2db320423bfc)
![image](https://github.com/user-attachments/assets/4dc48422-ccc4-476c-84cc-75e417b158c0)

#  Insights into Research Themes and Organizational Perception

![image](https://github.com/user-attachments/assets/fe2abff3-2c47-47f1-9736-b9ffb078c37c)

The bar chart showed on Figure 3 illustrates the sentiment score of research related to AI and
machine learning from 1985 to 2023. The y-axis represents the average sentiment, with values
ranging from 0 to 0.7, while the x-axis shows the years. Each bar corresponds to a specific year,
with the sentiment score labeled at the top of each bar.

![image](https://github.com/user-attachments/assets/6d3ecf2c-d821-4415-92ec-3624377ddcad)
![image](https://github.com/user-attachments/assets/db4a9c72-529f-4b9c-93e6-3db7f3f78f98)
![image](https://github.com/user-attachments/assets/d70b1ce7-0f4c-4407-bbe1-2e488529c8e9)

This distribution reinforces the positive perception identified earlier, where AI and machine
learning research, particularly in high-growth sectors like technology, transportation, and
science, has increasingly been seen as favorable. (Curry et al. 2021). The clustering of sentiment
around higher values suggests that organizations view these technologies as beneficial to
operational efficiency, innovation, and strategic planning. The significant frequency of neutral
sentiment scores aligns with the scientific rigor and objectivity often found in research
documents, where results are presented with caution and without emotional bias.


>[!NOTE]
>For the Part of Speech (POS) it was used only the following categories proper nouns (NNP, NNPS), technical terms (domain-specific), nouns (NN, NNS), and verbs (VB, VBD, VBG, VBN, VBP, VBZ) 

### Topics modeling and Table

![image](https://github.com/user-attachments/assets/7ef1e1dd-a956-45f1-920b-2fc7289f4817)

The table and topic modeling presents a series of topics, each represented by the top keywords that define its
thematic focus. These topics are likely the result of a topic modeling analysis, which identified
patterns and recurring themes across a collection of AI and machine learning research
documents. The terms within each topic provide insights into the specific areas of focus in the
research corpus. For example, Topic 1, which includes terms like "economic," "european," and
"political," suggests a focus on AI's impact within economic and policy-related contexts, while
Topic 3, with terms like "research," "ai," and "design," centers on core AI methodologies and
research practices. Other topics, such as Topic 7, with terms like "energy" and "security,"
indicate AI's application in areas like energy systems and cybersecurity.

| Topic 0        | Topic 1         | Topic 2           | Topic 3       | Topic 4       | Topic 5         | Topic 6       | Topic 7        |
|-----------------|----------------|-------------------|---------------|---------------|-----------------|---------------|----------------|
| search         | economic       | language          | research      | system        | early           | model         | energy         |
| algorithm      | european       | logic             | ai            | model         | surface         | method        | security       |
| linear         | political      | information       | design        | information   | showed          | performance   | channel        |
| case           | environmental  | reasoning         | development   | design        | late            | neural        | system         |
| tree           | impact         | representation    | digital       | order         | temperature     | algorithm     | connection     |
| solution       | article        | framework         | social        | control       | observed        | network       | edge           |
| random         | market         | theory            | work          | agent         | production      | machine       | wireless       |
| ai             | french         | semantic          | article       | user          | material        | deep          | performance    |
| optimal        | financial      | heidelberg        | human         | environment   | site            | optimization  | network        |
| finite         | role           | fuzzy             | management    | complex       | atlas           | training      | detection      |
| classical      | risk           | natural           | product       | framework     | age             | efficient     | timing         |
| distribution   | local          | order             | machine       | dynamic       | middle          | control       | communication  |

![image](https://github.com/user-attachments/assets/2f9ddc03-7738-4683-aeee-aea4bd8937ae)

 Figure 9 describes the yearly fluctuation in average sentiment across various
research topics. The y-axis represents average sentiment scores, while the x-axis shows the
timeline from 1985 to 2020. Each colored line corresponds to a different research topic.
Throughout the timeline, sentiment across all topics fluctuates, with notable variability in the
earlier years, particularly between 1985 and 2000, where sharp rises and falls are observed.
However, from 2000 onwards, the trends stabilize, and sentiment shows a gradual upward trend
across most topics, particularly from 2015 onward.

![image](https://github.com/user-attachments/assets/c3abd4bb-69ed-4b73-8883-88a6a6ea01f2)

The Correlation Matrix in Figure 11 visualizes the relationships between sentiment scores across
eight distinct AI research topics. The matrix is color-coded, with darker red shades indicating
higher positive correlations and lighter shades reflecting weaker or negative correlations. Strong
positive correlations, such as between Topics 1 and 7 (0.95), or Topics 0 and 3 (0.90), suggest
thematic or perceptual overlap, meaning that research in these areas tends to evoke similar
sentiment reactions. In contrast, weaker correlations, such as between Topic 4 and Topic 2
(0.47), indicate that sentiment perceptions in these areas diverge, possibly due to differences in
focus or complexity.

![image](https://github.com/user-attachments/assets/4a8fe48f-e3cd-4716-83ba-82d08840ccee)
![image](https://github.com/user-attachments/assets/b1e03337-5720-4e66-9cca-ee28dad8e4ee)

## Data
- Research about AI & business in France
    - Articles, research, books, and more founded on Scopus advanced searching, building a Dataset with all the metadata of different sources
    - URL: https://www.scopus.com/standard/marketing.uri

        ``` 
        ( TITLE-ABS-KEY ( ai ) OR ALL ( ai ) AND ALL ( business ) OR ALL ( organization ) OR ALL ( enterprise ) OR ALL ( work ) OR ALL ( profession ) OR ALL ( career ) OR ALL ( affair ) OR ALL ( occupation ) OR ALL ( report ) OR ALL ( market ) OR ALL ( invest ) OR ALL ( trade ) OR ALL ( industry ) OR ALL ( company ) OR ALL ( commerce ) OR ALL ( dealing ) OR ALL ( firm ) OR ALL ( purchase ) OR ALL ( survey ) OR ALL ( manager ) OR ALL ( manage ) OR ALL ( decision ) OR ALL ( digital ) OR ALL ( task ) OR ALL ( automate ) OR ALL ( processes ) OR ALL ( production ) OR ALL ( bot ) OR ALL ( data ) OR ALL ( selling ) OR ALL ( marketing ) OR ALL ( logistics ) OR ALL ( finance ) OR ALL ( fabricate ) OR ALL ( price ) OR ALL ( stock ) OR ALL ( network ) OR ALL ( resource ) OR ALL ( money ) OR ALL ( cash ) OR ALL ( credit ) OR ALL ( institution ) OR ALL ( smart ) OR ALL ( tech ) OR ALL ( case ) OR ALL ( trading ) OR ALL ( area ) OR ALL ( system ) AND NOT ALL ( medicine ) AND NOT ALL ( patients ) AND NOT ALL ( patients ) AND NOT ALL ( health ) AND NOT ALL ( disease ) AND NOT ALL ( covid-19 ) AND NOT ALL ( healthcare ) AND NOT ALL ( viral ) ) AND PUBYEAR > 1969 AND PUBYEAR < 2024 AND ( LIMIT-TO ( AFFILCOUNTRY , "France" ) ) AND ( LIMIT-TO ( SUBJAREA , "SOCI" ) OR LIMIT-TO ( SUBJAREA , "BUSI" ) OR LIMIT-TO ( SUBJAREA , "DECI" ) OR LIMIT-TO ( SUBJAREA , "MULT" ) OR LIMIT-TO ( SUBJAREA , "ECON" ) OR LIMIT-TO ( SUBJAREA , "PSYC" ) OR LIMIT-TO ( SUBJAREA , "ENGI" ) OR LIMIT-TO ( SUBJAREA , "MATH" ) OR LIMIT-TO ( SUBJAREA , "NEUR" ) OR LIMIT-TO ( SUBJAREA , "ENER" ) OR LIMIT-TO ( SUBJAREA , "ARTS" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )
        ```

- French Government AI related economic information
    - Cases, projects, regulations, and more related to AI in France, the source is the official site of the government, and the dataset created was web scrapped
    - URL: https://www.economie.gouv.fr/recherche-resultat?search_api_views_fulltext=IA&page=0
    - https://www.economie.gouv.fr/
- News of AI-related business, organization, economics and finance
    - The dataset created was web scrapped, applying different queries of searching into google search web scrapping tool defined on the folder "Websccrapping", results are compiled on a single dataset that contains all organizations and business AI-related results
- Enterprises_AI
    - Information about capital, composition, operations, and more about enterprises that lead the use and application of AI 
    - URL:     https://data.world/aurielle/the-essential-landscape-of-enterprise-a-i-companies/workspace/file?filename=EnterpriseAI.csv
- AI Tools
    - Tools based on different categories to automatize processes and optimize business, there are two sources, one official of French gouvernment datasets and another extracted from Kagle, in which it can be seen tools description and some metadata
    - URL GouvFR: https://www.data.gouv.fr/fr/datasets/4000-outils-ai/
    - URL Kagle: https://www.kaggle.com/datasets/yasirabdaali/740-ai-tools-for-everyone 
- Data Scientist vs size of Datasets
    - A group of 100 data scientists from France were interviewed between January 2016 and August 2016 to analyze the potential relationship between hardware and data set sizes. However, it is important to note that the sample size may not represent the entire population. Dataset obtained from Kagle
    - URL Kagle: https://www.kaggle.com/datasets/laurae2/data-scientists-vs-size-of-datasets/data
- Financial impact on CAC40
    - According to the results obtained from previous experiments on this project and guided by topic modeling structure, it has been selected specific companies that are related to AI use cases on their internal processes to analyze how the impact was received. The dataset built based on financial statements and open financial information 
    - URL: https://fr.finance.yahoo.com/             

>[!WARNING]
> It is necessary to install the libraries listed below and understand the requirements to make the hypothesis and Data Treatment

## Principal processes and requirements (AZURE, web scrapping, libraries, etc)

- Topic modeling documentation: https://github.com/piskvorky/gensim?tab=readme-ov-file

- Topic Modeling Gensim quick guide: https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html

- Creating Azure resource text analysis on Language Studio to obtain API key and endpoint for processing text analysis 
    
    Sample:
    https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/textanalytics/azure-ai-textanalytics/samples/sample_analyze_sentiment.py

- Creating virtual environment variables with Azure "text analysis" key and endpoint on a file .env and calling it on Python script as the following structure:
    
    - Load environment variables from the .env file, these would be the keys and endpoint for used API's - load_dotenv()
    - Access the environment variables

      ``` 
        endpoint = os.getenv("AZURE_TEXT_ANALYSIS_ENDPOINT")
        key = os.getenv("AZURE_TEXT_ANALYSIS_KEY")
      ``` 

- Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

      Dependence list: ['pybind11<2.6.2', 'psutil', "numpy>=1.10.0,<1.17 ; python_version=='2.7'", "numpy>=1.10.0 ; python_version>='3.5'"]     
      running bdist_wheel
      running build
      running build_ext
      Extra compilation arguments: ['/EHsc', '/openmp', '/O2', '/DVERSION_INFO=\\"2.1.1\\"']    
      building 'nmslib' extension

- Required python Libraries
    #### For Webscrapping
        pip install matplotlib
        pip install wordcloud
        pip install bs4
        pip install selenium
        pip install webdriver_manager
        pip install undetected_chromedriver
        pip install requests beautifulsoup4
        pip install googlesearch-python
        pip install streamlit

    #### For Topic Modeling 
        pip install --upgrade gensim
        pip install Pyro4
        pip install Sphinx
        pip install annoy
        pip install memory-profiler
        pip install nltk
        pip install nmslib (previous requirement of C++ development tools)
        pip install POT
        pip install scikit-learn
        pip install sphinx-gallery
        pip install sphinxcontrib-napoleon
        pip install sphinxcontrib-programoutput
        pip install statsmodels
        pip install testfixtures
        pip install spacy # For stop words
        python -m spacy download en_core_web_sm  # for English
        python -m spacy download fr_core_news_sm $ for French
        pip install gensim nltk pyLDAvis
        pip install gensim nltk matplotlib
        pip install Flask
        pip install TextBlob

    #### For Azure and other API's connection
        pip install azure-ai-textanalytics
        pip install azure-identity
        pip install python-dotenv

    ### For BERT transformers with pytorch
        pip install torch
        pip install transformers torch

    #### For better optimization with NumPy and OpenBLAS on the NLP with topic modeling
        pip install numpy --only-binary :numpy: numpy
        pip install numpy --no-binary numpy

>[!NOTE]
>References: disposed on the file 'references.txt'
