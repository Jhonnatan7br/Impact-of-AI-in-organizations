#### This repository is on construction
# To Solve

- Text analysis Message: Batch request contains too many records. Max 10 records are permitted. (Create a cycle to iterate)
  https://language.cognitive.azure.com/tryout/namedEntities 
- Topic Modeling with LSA, PLSA, LDA & lda2Vec (Also consider BERT)
https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
![image](https://github.com/Jhonnatan7br/Impact-of-AI-in-organizations/assets/104907786/bfd97b7b-2ee8-4d5e-8b6b-57ca89fc7d1e)


# Explanaiton of resources, process and requirements

Aim of subject and experiemnts diagram

### Data
- source: URL
    description
- source2: URL
    description

### Main conclusions

It could be made a BigData analysis of news with webscrapping, but it could be necesary more computational power, and optimal connection and latency to wifi and certain budget on Azure API of text analysis to analyze more data  

### Principal processes and requirements (AZURE, webscrapping, libraries, etc)

- Creating Azure resource text analysis on Language studio to obtain API key and enpoint for procesing text analysis 
    
    Sample:
    https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/textanalytics/azure-ai-textanalytics/samples/sample_analyze_sentiment.py

- Creating virtual environment variables with Azure "text analysis" key and enpoint on a file .env and calling it on python script as the following structure:
    
    Load environment variables from the .env file
    load_dotenv()
    Access the environment variables
    endpoint = os.getenv("AZURE_TEXT_ANALYSIS_ENDPOINT")
    key = os.getenv("AZURE_TEXT_ANALYSIS_KEY")    

- Required Libraries

    pip install matplotlib
    pip install wordcloud
    pip install bs4
    pip install selenium
    pip install webdriver_manager
    pip install undetected_chromedriver
    pip install requests beautifulsoup4
    pip install googlesearch-python

    pip install streamlit
    pip install azure-ai-textanalytics
    pip install azure-identity
    pip install python-dotenv

- 

### References
