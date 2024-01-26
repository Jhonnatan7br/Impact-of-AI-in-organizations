![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/d87a7990-b7c8-4769-a55e-9e028f2ca039)

### This repository is on construction

### To Solve or TO DO

- Text analysis Message: Batch request contains too many records. Max 10 records are permitted. (Create a cycle to iterate)
  https://language.cognitive.azure.com/tryout/namedEntities 
- Topic Modeling with LSA, PLSA, LDA & lda2Vec (Also consider BERT)
https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/bfd97b7b-2ee8-4d5e-8b6b-57ca89fc7d1e)
- Based on Topic Modeling results, construct a financial analysis of companies that appear are most related to AI 

# Explanaiton of resources, process and requirements

Aim of subjec, experiemnts & diagram

### Data
- Research about AI & business on France
    - Articles, researching, books, and more founded on scopus advanced searching, building a Dataset with all the metadata of different sources
    - URL: 
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
- Financial impact
    - According to the results obtained on previous experiments on this project and guided by topic modeling structure, it has been selected specific companies that are related to AI usecases on their internal processes to analyse how was the impact received. Dataset builded based on financial statements and open financial information 
    - URL: https://fr.finance.yahoo.com/             

### Main conclusions

It could be made a BigData analysis of news with webscrapping, but it could be necesary more computational power, and optimal connection and latency to wifi and certain budget on Azure API of text analysis to analyze more data  

### Principal processes and requirements (AZURE, webscrapping, libraries, etc)

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
    # For Webscrapping
    - pip install matplotlib
    - pip install wordcloud
    - pip install bs4
    - pip install selenium
    - pip install webdriver_manager
    - pip install undetected_chromedriver
    - pip install requests beautifulsoup4
    - pip install googlesearch-python
    - pip install streamlit

    # For Azure and other API's connection
    - pip install azure-ai-textanalytics
    - pip install azure-identity
    - pip install python-dotenv

    # For Topic Modeling 
    - pip install --upgrade gensim
    - pip install Pyro4
    - pip install Sphinx
    - pip install annoy
    - pip install memory-profiler
    - pip install nltk
    - pip install nmslib ***
     
    - pip install POT
    - pip install scikit-learn
    - pip install sphinx-gallery
    - pip install sphinxcontrib-napoleon
    - pip install sphinxcontrib-programoutput
    - pip install statsmodels
    - pip install testfixtures

    For better optimization with NumPy and OpenBLAS
    - pip install numpy --only-binary :numpy: numpy
    - pip install numpy --no-binary numpy
