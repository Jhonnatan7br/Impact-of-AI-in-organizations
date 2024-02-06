![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/d87a7990-b7c8-4769-a55e-9e028f2ca039)

### This repository is on construction

Dataset connections
![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/d06dacbf-888c-4844-8695-c020f5adcb66)


### To Solve or TO DO

- Text analysis Message: Batch request contains too many records. Max 10 records are permitted. (Create a cycle to iterate)
  https://language.cognitive.azure.com/tryout/namedEntities 

- Topic Modeling with LSA, PLSA, LDA & lda2Vec (Also consider BERT)

LDA Model: https://radimrehurek.com/gensim/models/ldamodel.html
https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05

- Based on Topic Modeling results, construct a financial analysis of companies that appear are most related to AI 

    Topics - Industries - Enterprises CAC40 - There is a correlation between the increase of research on certain fields and industries and the increase of value of companies on the same industry?

# Aim of subjec, experiemnts & diagram

First, it was trained and used an NLP (Natural Language Processing model to analyze research, articles, governmental and news information, there were used LDA, LSA and BERT machine learning models

![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/db46d0e8-0e3e-49c8-96d0-8c7764a8d91e)

EXTRACTED TOPICS TABLE

Taking into account that the impact and interest about use cases of AI is increasing, specially since 2018, it can be made the following graph to see this tendency (Scopus dataset)

![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/65f3156f-370b-4602-95f4-34311a4d13fc)

After that it was searched the relation between the topics founded and the data through statistical analysis

Understanding this tendency to grow and starting to analyze official information documented by the government of France through 93 cases (FRgov dataset), it was applied a keyword frecuency statistic to meassure in wich industries it exist more cases and if it was correlation between the results of Topic Modeling

![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/d92a6272-abaa-41f3-889f-85fd1f5ab996)

AI Tools dataset documented on the official framce Government databases 

![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/6e9acf04-ecc3-4fd2-b607-26242906ed57)


![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/7afdc492-65bf-478a-9bf7-2f7aa92111ff)

![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/ee44dd05-daa6-41a3-bae8-3180de5ded20)

XXX

# Explanaiton of resources, process and requirements

## Data
- Research about AI & business on France
    - Articles, researching, books, and more founded on scopus advanced searching, building a Dataset with all the metadata of different sources
    - URL: https://www.scopus.com/standard/marketing.uri

        ``` 
        ( TITLE-ABS-KEY ( ai ) AND ALL ( business ) OR ALL ( organization ) OR ALL ( enterprise ) OR ALL ( work ) OR ALL ( profession ) OR ALL ( career ) OR ALL ( affair ) OR ALL ( occupation ) OR ALL ( report ) OR ALL ( market ) OR ALL ( invest ) OR ALL ( trade ) OR ALL ( industry ) OR ALL ( company ) OR ALL ( commerce ) OR ALL ( dealing ) OR ALL ( firm ) OR ALL ( purchase ) OR ALL ( survey ) OR ALL ( manager ) OR ALL ( manage ) OR ALL ( decision ) OR ALL ( digital ) OR ALL ( task ) OR ALL ( automate ) OR ALL ( processes ) OR ALL ( production ) OR ALL ( bot ) OR ALL ( data ) OR ALL ( selling ) OR ALL ( marketing ) OR ALL ( logistics ) OR ALL ( finance ) OR ALL ( fabricate ) OR ALL ( price ) OR ALL ( stock ) OR ALL ( network ) OR ALL ( resource ) OR ALL ( money ) OR ALL ( cash ) OR ALL ( credit ) OR ALL ( institution ) OR ALL ( smart ) OR ALL ( tech ) OR ALL ( case ) OR ALL ( trading ) OR ALL ( area ) OR ALL ( system ) ) AND ( LIMIT-TO ( AFFILCOUNTRY , "France" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) ) 
        ```

- French Gouvernment AI related economic information
    - Cases, projects, regulations and more related to AI on France, the source is the official site of government, and the dataset created was webscrapped
    - URL: https://www.economie.gouv.fr/recherche-resultat?search_api_views_fulltext=IA&page=0
    - https://www.economie.gouv.fr/
- News of AI related business, organization, economic and finance
    - The dataset created was webscrapped, applying different queries of searching into googlesearch webscrapping tool defined on the folder "Websccrapping", results are compiled on a single dataset that contains all organizations and business AI related results
- Enterprises_AI
    - Information about capital, composition, operations and more about enterprises that lead the use and application of AI 
    - URL:     https://data.world/aurielle/the-essential-landscape-of-enterprise-a-i-companies/workspace/file?filename=EnterpriseAI.csv
- AI Tools
    - Tools based on different categories in order to automatize processes and optimize business, there are two sources, one official of french gouvernment datasets and another extracted from Kagle, in wich it can be see tools description and some metadata
    - URL GouvFR: https://www.data.gouv.fr/fr/datasets/4000-outils-ai/
    - URL Kagle: https://www.kaggle.com/datasets/yasirabdaali/740-ai-tools-for-everyone 
- Data Scientist vs size of Datasets
    - A group of 100 data scientists from France were interviewed between January 2016 and August 2016 to analyze the potential relationship between hardware and data set sizes. However, it is important to note that the sample size may not be representative of the entire population. Dataset obtained from Kagle
    - URL Kagle: https://www.kaggle.com/datasets/laurae2/data-scientists-vs-size-of-datasets/data
- Financial impact on CAC40
    - According to the results obtained on previous experiments on this project and guided by topic modeling structure, it has been selected specific companies that are related to AI usecases on their internal processes to analyse how was the impact received. Dataset builded based on financial statements and open financial information 
    - URL: https://fr.finance.yahoo.com/             

## Main conclusions

It could be made a BigData analysis of news with webscrapping, but it could be necesary more computational power, and optimal connection and latency to wifi and certain budget on Azure API of text analysis to analyze more data  

## Principal processes and requirements (AZURE, webscrapping, libraries, etc)

- Topic modeling documentation: https://github.com/piskvorky/gensim?tab=readme-ov-file

- Topic Modeling Gensim quick guide: https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html

- Creating Azure resource text analysis on Language studio to obtain API key and enpoint for procesing text analysis 
    
    Sample:
    https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/textanalytics/azure-ai-textanalytics/samples/sample_analyze_sentiment.py

- Creating virtual environment variables with Azure "text analysis" key and enpoint on a file .env and calling it on python script as the following structure:
    
    - Load environment variables from the .env file, these would be the keys and endpoint for used API's - load_dotenv()
    - Access the environment variables
        endpoint = os.getenv("AZURE_TEXT_ANALYSIS_ENDPOINT")
        key = os.getenv("AZURE_TEXT_ANALYSIS_KEY")    
        
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

    #### For Azure and other API's connection
        pip install azure-ai-textanalytics
        pip install azure-identity
        pip install python-dotenv

    ### For BERT transformers with pytorch
        pip install torch
        pip install transformers torch

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

    #### For better optimization with NumPy and OpenBLAS on the NLP with topic modeling
        pip install numpy --only-binary :numpy: numpy
        pip install numpy --no-binary numpy

>[!NOTE]
>References: disposed on the file 'references.txt'
