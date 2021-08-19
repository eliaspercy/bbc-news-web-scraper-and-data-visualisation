## Python version 3.9
"""
PREREQUISITES

The code was written on a Windows 10 operating system.
The version of Python used was 3.9

Pre-requisite libraries/modules:
    - requests
    - pandas
    - numpy
    - lxml  
    - bs4
    - nltk
    - sklearn

Each of the above and be installed using pip in the terminal, i.e.:
    - pip install [module]
    
The program itself can simply be run in the terminal, i.e.:
    - python 000810185.py
"""

# Main imports
from typing import Set, Tuple, List, Dict, Union, Generator
from bs4 import BeautifulSoup, SoupStrainer # type: ignore
import requests as req
import pandas as pd # type: ignore
import numpy as np
import time
import os
np.random.seed(42)

# ---- PARAMETERS ------ #
VERBOSE: bool = True

# Problem 1 (Web scraper)
PAGE_LIMIT: int = 30
NUM_WORKERS: int = 100

# Problem 3 (Word2vec)
WINDOW_SIZE: int = 2
VECTOR_DEPTH: int = 300
NEG_SAMPLE_SIZE: int = 5
NUM_EPOCHS: int = 2
LEARN_RATE: float = 0.01
DIST_ALPHA: float = 0.75
COLLECT_ALL: bool = True

# --------------------- #


##Problem 1
"""
To collect the articles, I wrote a generator function that takes a keyword as 
an argument and makes consecutive requests to the BBC search URL corresponding 
with this argument page by page: before moving on to the next page and 
repeating the following process, after making the request to each search page 
my algorithm will first iterate through the links within said search page and 
collect any *valid* links it finds, then ensures that the text content in the 
webpage corresponding with each link contains the keyword to some degree, 
before *yielding* this webpage content. Once all pages have been exhausted, 
the generator will return `None`, thereby halting. The webpage content for 
each keyword is collected within a list, and the `itertools` library (namely, 
the functions `islice()` and `takewhile()` have been used, building from my 
knowledge of lazily evaluated programming languages, to ensure that these lists 
do not exceed a length of 100 or that they stop collecting immediately upon 
receiving `None`). The lists containing the aforementioned webpage content for 
each keyword are stored in a dictionary, where the keys are the keywords and 
the values are the lists. This data structure was selected due to its ease of 
usage and convenience, and it lent naturally to the following question.

Worth noting also that my function incorporates error checking when making 
requests. This part of the program deals extensively with making HTTP requests
using the requests library, and so the potential of network partitions must be
considered, as they are generally unavoidable. Thus, I implemented a method for
containing any request made, which utilises a `while` loop and a `try-except` 
statement that, upon making an invalid request (i.e., one during which some
network partition occurred), the loop is restarted and the request is 
re-attempted. This will continue until the request is successful (that is, no 
requests exception is raised).

To enhance the efficiency of the web-scraping, asynchronous methods were used.
That is, the web pages for each keyword are collected in parallel. The number 
of "workers" used here dictates the speed, and this is alterable as an 
adjustable parameter.

There are two adjustable parameters for this problem, configurable above. These
are the number of workers (mentioned previously), and the page limit - this 
places an upper bound on the number of pages the BBC search may visit, in order
to prevent excessive runtimes. The program will automatically stop searching 
for articles when no more are available, but this is an extra measure for 
extreme circumstances. By default, it is set to 50.

Also note that in order to guarantee article relevence, each article is checked
to ensure it actually contains the corresponding keyword so some extent (i.e., 
the word actually appears in the article).
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, takewhile, count

KEYWORDS:        str = './keywords.xlsx' 
BBC_URL:         str = 'https://www.bbc.co.uk/news/'
BBC_URL_ALT:     str = 'http://www.bbc.co.uk/news/'
BBC_URL_OLD:     str = 'http://news.bbc.co.uk/'
BBC_URL_OLD_ALT: str = 'https://news.bbc.co.uk/'
SEARCH_URL:      str = 'https://www.bbc.co.uk/search'
LINKS_CLASS:     str = 'ssrcss-10ivm7i-PromoLink e1f5wbog5'
HEADERS: Dict[str, str] = {
    'user-agent': 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'
}
BBC_URLS: Tuple[str, ...] = (   
    BBC_URL, BBC_URL_ALT, 
    BBC_URL_OLD, BBC_URL_OLD_ALT
)


def collect_keywords(kw_file: str
                     ) -> List[str]:
    """ 
    Read in the .xlsx file containing the keywords. Store the keywords in a 
    Python list (so as to retain order).
    """
    try:
        keywords_df = pd.read_excel(kw_file, sheet_name=0)
        keywords = list(keywords_df['Keywords'])
    except FileNotFoundError:
        raise Exception("[ERROR] The keywords file was not found! This is "
                        "required for the program to run.")
    except KeyError:
        raise Exception("[ERROR] The keywords file does not contain a "
                        "'Keywords' column! The file must contain this column "
                        "followed by a list of keywords.")
    return keywords


class WebScraper:
    """ 
    Class containing the methods for interrogating the BBC search engine and 
    collecting all relevent articles associated with the inputted keywords. A
    requests Session is initialised at the construction of this class to allow
    for slightly more efficient request making, and the webpages for each 
    keyword are gathered in parallel (i.e., asynchronoushly).
    """
    
    keywords: List[str]
    session: req.Session
    
    def __init__(self, 
                 keywords: List[str]
                 ) -> None:
        self.keywords = keywords
        self.session = req.Session()
        self.session.headers = HEADERS
    
    @staticmethod
    def verify(link: str
               ) -> bool:
        """ 
        Verify any collected link is acceptable by criteria
        """
        
        # Ignore any /live/ or /av/ articles as they aren't proper articles
        if any([path in link for path in ("/live/", "/sport1/", "/av/")]):
            return False
        
        # Ensure the link corresponds with a valid BBC News article.
        return any([link.startswith(prefix) for prefix in BBC_URLS])

    def get_request(self: 'WebScraper', 
                    url: str, 
                    params: Union[dict, None] = None
                    ) -> req.Response:
        """ 
        Function for safely making get requests. This program entails making 
        over 1000 requests to the BBC domain, thus it's indeed possible for a
        network partition to occur during this process. As a response, I 
        included this function that will first try to make the request and if
        a network partition occurs (i.e., a requests exception is raised), then
        the function will 'sleep' for 1 second before retrying.
        """
        res = None
        while True:
            try: 
                res = self.session.get(url, params=params)
                break
            except req.RequestException: 
                # If a requests exception is raised, then some kind of HTTP 
                # error occured, so wait for one second before re-attempting 
                # the request
                time.sleep(1)
                continue
        return res
                
    def get_links(self: 'WebScraper', 
                  keyword: str
                  ) -> Generator[req.Response, None, None]:
        """
        Generator for collecting the web-page content for a given keyword. For 
        a given keyword, starting at page 1 this function will make a request
        to the corresponding BBC search page, and collect all the links for
        the query result after validating them. To validate the links, the 
        method checks first if the link is a valid BBC new link (i.e., it 
        begins with the correct prefix, and it isn't a video articke)
        """
        print(f"Collecting articles for the keyword '{keyword}'...")
        
        # Create strainer that only searched for links with the corresponding 
        # class specified in the constant LINKS_CLASS
        only_links = SoupStrainer(
            'a', {'class': LINKS_CLASS}
        )
        parameters = {'q': keyword}
        
        # Iterate through the pages of the search
        for i in count(1):

            # Stop when the page limit has been reached
            if i > PAGE_LIMIT:
                return None
            
            # for keyword in keyword_synonyms:
            parameters['page'] = i
            res = self.get_request(SEARCH_URL, parameters)
            links = {
                link['href'] 
                for link in BeautifulSoup(
                    res.text, 'lxml', 
                    parse_only=only_links
                ).find_all('a', href=True) 
                if self.verify(link['href'])
            }
            
            for link in links:
                this = self.get_request(link)
                if keyword.lower() in this.text.lower():
                    yield this

    def collect_webpages(self, keyword: str) -> Dict[str, List[req.Response]]:
        """
        Collect the top-100 relavent articles for the keywords using the notion
        of infinite (lazily-evaluated) lists, with takewhile and islice. 
        """
        collected = {
            keyword: list(islice(
                takewhile(lambda x: x is not None, self.get_links(keyword)), 
                100
            ))
        }
        print(f"Found {len(collected[keyword])} articles for the keyword "
              f"'{keyword}'.")
        return collected

    async def initialiser_crawler(self) -> Dict[str, List[req.Response]]:
        """ 
        Initialise and instantiate the asynchronous web crawling.
        """
        web_pages = {}
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            try:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(exe, self.collect_webpages, keyword)
                    for keyword in self.keywords 
                ]
                for res in await asyncio.gather(*tasks):
                    web_pages.update(res)
            except KeyboardInterrupt:
                loop.close()
                raise KeyboardInterrupt
        return web_pages
    
    def crawl(self) -> Dict[str, List[req.Response]]:
        """ 
        Set up a loop and begin crawling.
        """
        loop = asyncio.get_event_loop()
        try:
            web_pages = loop.run_until_complete(
                asyncio.ensure_future(self.initialiser_crawler())
            )
        except KeyboardInterrupt:
            loop.close()
            raise KeyboardInterrupt
        return web_pages


def problem1() -> Dict[str, List[req.Response]]:
    """ 
    Container function for the methods comprising problem 1
    """
    keywords = collect_keywords(KEYWORDS)
    if len(keywords) == 0:
        raise Exception("[ERROR] No keywords were found in the keyword file!")
    web_scraper = WebScraper(keywords) 
    web_pages = web_scraper.crawl()
    return web_pages


##Problem 2
""" 
The following section comprises the functions I have implemented for problem 2 
of the coursework. That is, the following functions serve to filter through the
articles collected during problem 1 and process them, in order to ultimately
save them locally in text files. I decided to save them in text files where 
sentences are separated by a new line (i.e., by '\n') as this allows for easy
processing of them later on. I also consciously decided to keey some of the 
text associated with the reccommened articles at the bottom of some of the 
articles as I decided that they had lexical value (because they were related to
the article being collected). Moreover, the text files are saved in separate 
folders for each keyword, to reduce clutter, and these folders will reside in 
a superfolder named "articles". The filenames are the keyword followed by a 
number distinguishing it from the other articles by the order they are saved.

There are 8 functions for processing the BBC news articles - two for handling
two distinct formats that appear in webpages that utilise the current URL 
format, and 6 for handling variations of articles that appear under the older
URL format. A 'domino' like effect dictates how the articles are processed. 
That is, it is first established whether or not they are new or old articles, 
and processing is then attempted from there. If an AttributeError or IndexError
is raised, then this implies that the format is not suitable for this 
particular function, and thus a different method is attempted. 

I decided to partition the functions in this way for the sake of clarity. 
Attempting to handle all articles in a single or pair of functions leads to 
rather incomprehensible code, whereas having a distinct function for each of 
the possible formats is more clear and readable. Moreover, this particular 
manner of separating the functions allows for further scalability, in that it 
is much easier to simply add an additional function if more formats neeed to 
be handled in the future. 

"""
from collections import namedtuple

FORMAT: str = 'utf-8'
SAVED_ARTICLES_PATH: str = os.path.join("articles")

article_content = namedtuple("article_content", "head body")


def tidy_string(s: str
                ) -> str:
    """ 
    Function for cleaning all strings collected from the articles. This entails
    encoding the string into ascii then decoding it into utf-8 in order to 
    remove unwanted symbols.
    """
    s = s.encode('ascii', errors='ignore').decode(FORMAT)
    s = s.replace("\r", "").replace("\t", "").replace('\n', '') 
    return s


def collect_body(body_area: BeautifulSoup
                 ) -> List[str]:
    """ 
    Function for collecting the body of an article via the <p> tags. This 
    function mainly serves to avoid repeated code in the functions below.
    """
    body = []
    for p in body_area.find_all("p"):
        body += [
            tidy_string(t) for t in p.text.split('\n') if tidy_string(t) != ""
        ]
    return body


def collect_content_old_alt_6(web_page: req.Response
                              ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article_area = this_soup.find('td', {'width': '629', 'valign': 'top'})
        article = article_area.find_all('table', {'width': '629'})[1]
        head = article.find('div', {'class': 'mxb'}).text
        head = tidy_string(head)
        body_area = article.find('td', {'width': '416', 'valign': 'top'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        print("Unable to handle the format of the following URL: ")
        print(web_page.url)
        return None


def collect_content_old_alt_5(web_page: req.Response
                              ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article_area = this_soup.find('td', {'width': '629', 'valign': 'top'})
        article = article_area.find_all('table', {'width': '629'})[1]
        head = article.find('div', {'class': 'sh'}).text
        head = tidy_string(head)
        body_area = article.find('div', {'class': 'storybody'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_alt_6(web_page)
        

def collect_content_old_alt_4(web_page: req.Response
                              ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article = this_soup.find('td', {'width': '328', 'valign': 'top'})
        head = " ".join([h.find('b').text for h in article.find_all(
            'font', {'size': '5'})]
        )
        head = head.replace("\r", "").replace("\t", "").replace('\n', '') 
        body_area = article.find('font', {'size': '2'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_alt_5(web_page)


def collect_content_old_alt_3(web_page: req.Response
                              ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        try:
            head = this_soup.find('div', {'class': 'headlinestory'}).text
        except AttributeError:
            head = this_soup.find('b', {'class': 'headlinestory'}).text
        head = tidy_string(head)
        body_area = this_soup.find('div', {'class': 'bodytext'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_alt_4(web_page)


def collect_content_old_alt_2(web_page: req.Response
                              ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article_area = this_soup.find('td', {'width': '629', 'valign': 'top'})
        article = article_area.find_all('table', {'width': '629'})[1]
        head = article.find('div', {'class': 'sh'}).text
        head = tidy_string(head)
        body_area = article.find('font', {'size': 2})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_alt_3(web_page)


def collect_content_old_1(web_page: req.Response
                          ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article = this_soup.find('table', {'class': 'storycontent'})
        head = article.find('h1').text
        head = tidy_string(head) 
        body_area = article.find('div', {'class': 'storybody'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_alt_2(web_page)


def collect_content_old_0(web_page: req.Response
                        ) -> Union[article_content, None]:
    """ 
    Function for handling news articles in the older format.
    """
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article = this_soup.find('table', {'class': 'storycontent'})
        head = article.find('h1').text
        head = tidy_string(head)
        body_area = article.find('td', {'class': 'storybody'})
        body = collect_body(body_area)
        return article_content(head, body)
    except (AttributeError, IndexError):
        return collect_content_old_1(web_page)


def collect_content_alt(web_page: req.Response
                        ) -> Union[article_content, None]:
    try:
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        article = this_soup.find('div', {'id': 'main-content'})
        head = article.find('h1', {"class": "story-header"}).text
        head = tidy_string(head)
        body = collect_body(article)
        return article_content(head, body)
    except (AttributeError, IndexError):
        print("Unable to handle the format of the following URL: ")
        print(web_page.url)


def collect_content(web_page: req.Response
                    ) -> Union[None, article_content]:
    """ 
    Function for utilising BeautifulSoup for collecting and decoding the
    content of a webpage. This takes in the raw request obtained in the prior
    section and extracts the valuable article detail from within it, and 
    returns this in the form of a namedtuple (with the header/title of the 
    article beside the body).
    A try-except statement has been used in order to handle the varying formats
    of the BBC news articles, creating a domino-effect. Moreover, articles are
    first checked to see if they are "old", that is that they are prefix by the
    outdated BBC news URL, and these are handled separately in similar 
    functions above this one.
    """
    try:
        
        # Check if the webpage begins with the older BBC URL - in which case,
        # use a different function to handle the article differently
        if web_page.url.startswith(BBC_URL_OLD):
            return collect_content_old_0(web_page)
        
        # Extract the article content using beautiful soup
        this_soup = BeautifulSoup(web_page.text, 'lxml')
        
        # Find the area of the content corresponding with the article - this is
        # the valuable information we want to keep
        article = this_soup.find('article', {})
        
        # Find and tidy the title of the article
        head = article.find('h1', {"id": "main-heading"}).text
        head = tidy_string(head)
        
        # Collect the article body and return a namedtuple containing the title
        # as a string and the body as a list of strings
        body = collect_body(article)
        return article_content(head, body)
    
    except (AttributeError, IndexError):
        # If an attribute or index error is raised, then the format of the 
        # current article must be handled differently, so use an alternative
        # function. This will continue in a cascading fashion until a suitable
        # function handles the format.
        return collect_content_alt(web_page)


def write_to_file(content: Union[article_content, None],
                  save_path: str,
                  keyword:   str,
                  num_file:  int
                  ) -> None:
    """
    Function for writing the decoded content of a news article to a .txt file.
    """
    if content is None: 
        return
    
    # Establish the filename for the text file, which will have the form
    # "[keyword]_[num].txt"
    file_name = f"{keyword.replace(' ', '_')}_{num_file:02d}"
    with open(f"{save_path}/{file_name}.txt", 'w') as new_article_txt:    
        
        # Write the head at the top of the text file
        try:    
            new_article_txt.write(f"{content.head}\n\n")
        except UnicodeEncodeError: 
            new_article_txt.write(
                f"{content.head.encode(FORMAT, 'replace')}\n\n"
            )
            
        # Write the article body under the heading, separating each sentence
        # with a single newline
        for line in content.body:
            try:
                new_article_txt.write(f"{line}\n")
            except UnicodeEncodeError:
                new_article_txt.write(f"{line.encode(FORMAT, 'replace')}\n")
                

def save_bbc_articles(web_pages: Dict[str, List[req.Response]],
                      save_path: str = SAVED_ARTICLES_PATH
                      ) -> None:
    """ 
    Function for initialising the decoding of the webpages and, and subsequent
    storing these webpages as text files.
    """
    
    # Make a directory called "articles" if one doesn't already exist
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
        
    # Iterate through the keywords and process the raw webpages obtained 
    for keyword in web_pages:
        print(f"Tidying and saving articles collected for the keyword "
              f"'{keyword }'...")
        kw_path = os.path.join(save_path, keyword)
        
        # Create a directory for the keyword if one doesn't exist
        if not os.path.exists(kw_path): 
            os.mkdir(kw_path)
        
        for num, page in enumerate(web_pages[keyword]):
            write_to_file(collect_content(page), kw_path, keyword, num)
            

def problem2(web_pages: Dict[str, List[req.Response]]) -> None:
    """ 
    Container function for the methods comprising problem 2
    """
    save_bbc_articles(web_pages)


##Problem 3
"""
The following section of code entails my implementation of an algorithm for 
calculating the semantic distance between specified keywords. The algorithm I
decided to implement is 'word2vec' - I settled on this one because it is a 
state of the art natural language processing algorithm for generating word 
embeddings, and in particular it lends naturally to the notion of semantic 
distance due to the fact that the semantic relationship between all word 
vectors is defined by the 'cosine similarity' between them.

Before the algorithm can correctly and effectively be executed, various 
preprocessing and data cleaning methods are necessary. This includes the 
removal of stop words - that is, words that have a disproportionately large
presence in the dataset yet provide little to no lexical value (such as 'the',
'a', and so on). Additionally, lemmatisation and word tokenisation are 
important stages of natural languages processing data preparation, and these
are implemented here. Please inspect the code and the corresponding comments
for more detail regarding the various preprocessing methods.

"""
import nltk # type: ignore
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from nltk import word_tokenize
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations
from collections import namedtuple

nltk.download('wordnet')
nltk.download('punkt')
Matrices = namedtuple("Matrices", "embedding context")
DISTANCE_XLSX: str = os.path.join("distance.xlsx")

# The following are stop-words that are more expansive than those available
# in NLTK.
STOPWORDS: Set[str] = {"-", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "ha", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"}


def process_article(sentences: List[Dict[str, str]],
                    article: str,
                    keyword: str,
                    collect_all: bool
                    ) -> List[Dict[str, str]]:
    """ 
    Process each article by collecting all of the sentences from it that 
    contain the keyword. This entails converting all characters in each of 
    these sentences to lowercase, so that the number of unique words is 
    decreased and thus the accuracy of the model downstream is increased.
    """
    with open(article, 'r') as txt:
        for line in txt.read().split('\n'):
            if collect_all or keyword.lower() in line.lower():
                sentences.append({
                    "sentence": line,
                    "keyword": keyword
                })
        
    return sentences


def collect_article_text(directory: str,
                         sentences: List[Dict[str, str]],
                         keyword: str,
                         collect_all: bool,
                         ) -> List[Dict[str, str]]:
    """ 
    Function for reading in each of the articles and collecting the relevent 
    sentences within them,. This data is stored in a Pandas DataFrame, which
    allows for more convenient analysis and visualisation of the data.
    """
    articles = (os.path.join(directory, article)
                for article in os.listdir(directory)
                if article.endswith(".txt"))
    for article in articles:
        sentences = process_article(sentences, article, keyword, collect_all)
    return sentences


def collect_keyword_dataframe(collect_all: bool = False,
                              path: str = SAVED_ARTICLES_PATH
                              ) -> pd.DataFrame:
    """ 
    Read the the article content - save as a pandas DataFrame to ensure more
    convenience with regards to visualisations.
    """
    sentences: List[Dict[str, str]] = []
    keywords = os.listdir(path)
    for keyword in keywords: 
        kw_directory = os.path.join(path, keyword)
        sentences = collect_article_text(
            kw_directory, sentences, keyword, collect_all
        )
    keyword_df = pd.DataFrame(sentences)
    return keyword_df


def separate_keywords_corpus(df: pd.DataFrame
                             ) -> Tuple[List[str], ...]:
    """ 
    Separate the keywords from the articles from the initial DataFrame so that
    the word2vec model can utilise them.
    """
    keywords = list(df['keyword'].unique())
    sentences = list(df['sentence'])
    return keywords, sentences


def subsample_frequent_words(corpus: List[str],
                             keywords: Set[str],
                             rate: float = 0.001,
                             threshold: float = 0.25
                             ) -> Set[str]:
    """
    Method for deriving a set of 'stop words' that appear frequently in the 
    text, so can be safely ignored by the algorithm as they offer little
    semantic insight. By trial-and-error, a threshold of 0.25 was shown to
    effectively collect stop-words, with some intervention to prevent keywords
    or likely-to-be valuable words being collected.
    """
    
    # Add some frequently occuring words that are likely to be insightful
    for word in ('security', 'attack', 'computer', 'data'):
        keywords.add(word)
    
    def sampling_func(w: str
                      ) -> bool:
        x = complete_vocab.count(w)/vocab_size
        return (np.sqrt(x/rate)+1)*rate/x < threshold
    
    complete_vocab = list(
        word for text in corpus for word in text if word not in keywords
    )
    unique_vocab = list(set(complete_vocab))
    vocab_size = len(complete_vocab)
    return STOPWORDS.union({w for w in unique_vocab if sampling_func(w)})
    

@dataclass
class W2VPreprocessor:
    """ 
    Class for preparing the data for the Word2Vec algorithm, which is the 
    algorithm I have selected for calculating the semantic distances between 
    the keywords. The preprocessing entails cleaning the inputted sentences 
    such that all words are in lower case, the sentences are tokensied and 
    have stop words removed, and various other methods. The purpose is to 
    prepare the data such that it is usuable by the word2vec model.
    """

    keywords:       List[str]
    corpus:         List[str]
    case_fold:      bool = True
    special_syms:   bool = True
    tokenise:       bool = True
    del_stop_words: bool = True
    stem:           bool = False
    lemmatise:      bool = True
    dist_alpha:     float = 0.75
    spaces:         Dict[str, str] = field(init=False)
    
    def __post_init__(self):
        self.spaces = {
            kw.lower(): kw.lower().replace(" ", "_") 
            for kw in self.keywords if " " in kw
        }
        
    def handle_kw_phrases(self, text: str
                          ) -> str:
        """
        Convert the spaces in any keyword with '_', so they can be treated as a 
        single word by the algorithm. 
        """
        for kw in self.spaces:
            if kw in text:
                text = text.replace(kw, self.spaces[kw])
        return text
    
    def depluralise_keywords(self, text: str
                             ) -> str:
        """ 
        Convert all occurences of any keyword as a plural into its singular 
        equivalent.
        """
        for kw in self.target_words:
            if kw+'s' in text:
                text = text.replace(kw+'s', kw)
        return text

    @staticmethod
    def handle_special_symbols(text: str
                               ) -> str:
        """
        Remove all "special" characters since unlikely to provide semantic 
        insight. Note that '_' is not removed, as this is used for the keywords
        with blank spaces.
        """
        valid_special_symbols = {' ', '_'}

        def criteria(c: str
                     ) -> str:
            return c if c.isalnum() or c in valid_special_symbols else ' '

        return ''.join(criteria(c) for c in list(text))

    def handle_stop_words(self,
                          text: str,
                          stop_words: Set[str]
                          ) -> Union[str, List[str]]:
        """ 
        Remove high frequency words that contain minimal semantic value.
        """
        if not self.tokenise:
            return ' '.join(
                w for w in word_tokenize(text) if w not in stop_words
            )
        return [w for w in text if w not in stop_words]
    
    def stemming(self,
                 text: str
                 ) -> Union[str, List[str]]:
        """
        Apply stemming to the text using the Lancaster stemmer, reducing all 
        words to a prefix in order to mitigate unhelpful variation in corpus.
        """
        stemmer = LancasterStemmer()

        def stem_sans_kw(w: str
                         ) -> str:
            return ( 
                stemmer.stem(w) if w not in self.target_words else w
            )
        
        if not self.tokenise:
            return ' '.join(
                stem_sans_kw(w) for w in word_tokenize(text)
            )
        return [stem_sans_kw(w) for w in text]

    def lemmatisation(self, 
                      text: str
                      ) -> Union[str, List[str]]:
        """
        Treat all "inflections" identically for the sake of the model. Similar 
        to stemming, except deriving semantic root instead of simple prefix. 
        """
        lemmatiser = WordNetLemmatizer()

        def lemma_sans_kw(w: str
                          ) -> str:
            return (
                lemmatiser.lemmatize(w) if w not in self.target_words else w
            )
        
        if not self.tokenise:
            return ' '.join(
                lemma_sans_kw(w) for w in word_tokenize(text)
            )
        return [lemma_sans_kw(w) for w in text]

    def tidy_text(self, text: List[str],
                  ) -> List[str]:
        """ 
        Container method for applying the transformations to the text.
        """
        
        if self.case_fold:
            text = list(map(lambda t: t.lower(), text))
                
        if self.special_syms:
            text = list(map(self.handle_special_symbols, text))
            
        text = list(map(self.handle_kw_phrases, text))
        text = list(map(self.depluralise_keywords, text))

        if self.tokenise:
            text = list(map(word_tokenize, text))
            
        if self.stem:
            text = list(map(self.stemming, text))
                
        if self.lemmatise:
            text = list(map(self.lemmatisation, text))
        
        if self.del_stop_words:
            stop_words = subsample_frequent_words(text, set(self.target_words))
            text = list(map(
                lambda t: self.handle_stop_words(t, stop_words), text
            ))
        
        return text

    @cached_property
    def preprocess_corpus(self) -> List[str]:
        """ 
        Cache the preprocessed corpus as an attribute of the class.
        """
        return self.tidy_text(self.corpus)
    
    @cached_property
    def target_words(self) -> List[str]:
        """ 
        A list of the "target words" - which are the key words but with the
        filtering/preprocessing applied.
        """
        return list(map(
            lambda w: self.spaces[w.lower()] 
            if w.lower() in self.spaces else w.lower(), 
            self.keywords
        )) 
    
    @cached_property
    def original2target(self) -> Dict[str, str]: 
        """ 
        Inverse of the method below.
        """
        return {
            self.keywords[i]: self.target_words[i]
            for i in range(len(self.keywords))
        }

    @cached_property
    def target2original(self) -> Dict[str, str]: 
        """ 
        Dictionary that maps the original key words onto their "encoded" 
        equivalent (i.e., with lower case, with spaces replaced with '_')
        """
        return {
            self.target_words[i]: self.keywords[i] 
            for i in range(len(self.keywords))
        }
    
    @cached_property
    def vocabulary(self) -> np.ndarray:
        """ 
        An array of the unique words from the corpus.
        """
        return np.array(
            list(set(word for text in self.preprocess_corpus for word in text))
        )
    
    @cached_property
    def vocab_size(self) -> int:
        """ 
        The number of unique words in the corpus.
        """
        return len(self.vocabulary)


def normalise_embedding(vector: np.ndarray
                        ) -> np.ndarray:
    """
    Function for normalising the vectors
    """
    norm = np.linalg.norm(vector)
    return vector/norm
    
    
def get_noise_distribution(corpus: List[str],
                           vocabulary: np.ndarray,
                           dist_alpha: float
                           ) -> List[int]:
    """ 
    Noise distribution for use in obtaining the negative samples (i.e., 
    dictates the likelihood of a word being chosen based on frequency), 
    with self.dist_alpha as a hyper-parameter
    """
    all_words = [word for text in corpus for word in text]
    arr = np.array(list(map(
        lambda x: all_words.count(x)**dist_alpha, vocabulary
    )))
    return arr/arr.sum()  # frequencies, normalised, in order of vocabulary

    
def one_hot_vocab_encoding(w2vp: W2VPreprocessor   
                           ) -> Dict[str, np.ndarray]:
    """
    Assign a unique oneHotEncoding to each distinct word within the corpus; 
    store within a dictionary. For convenience, instead of constructing actual
    one-hot encoding vectors, an integer representing the index of the one hot
    encoding is obtained for each word. This allows for more easy interaction 
    with and manipulation of the embedding matrices.
    """
    return {
        w: i for i, w in enumerate(w2vp.vocabulary)
    }


def get_matrices(dimensions: Tuple[int, int]
                 ) -> Matrices:
    """ 
    Initialise the embedding and context matrices
    """
    embedding_matrix = np.random.uniform(-1, 1, dimensions)
    context_matrix = np.random.uniform(-1, 1, dimensions)
    return Matrices(embedding_matrix, context_matrix)
    

def obtain_skipgram_dataset(corpus: list,
                            window_size: int,
                            ) -> pd.DataFrame:
    """
    Transform the relevant data into a pandas DataFrame for use in the model,
    where every vectorised word is accompanied by a context word. This follows
    the skipgram approach.
    """
    data = []
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            lower, upper = idx - window_size, idx + window_size + 1
            
            # Find neighbours, exclude centre word (even if neighbour).
            neighbours = list(filter(
                lambda w: w != word, islice(sentence, (lower>0)*lower, upper)
            ))
            
            # Add all instances of centre word and its neighbours to data.
            data += [
                {'centre_word': word, 'context_word': neighbour}   
                for neighbour in neighbours 
            ]

    return pd.DataFrame(data)


def negative_sampling(data: pd.DataFrame,
                      vocab: np.ndarray,
                      noise_distribution: list,
                      neg_sample_size: int
                      ) -> pd.DataFrame:
    """ 
    Apply negative sampling to the dataset, such that it is not only composed
    of words and their neighbours, but also includes words with words who do
    not appear as neighbours in the corpus. This is done via random sampling.
    """
    
    def samples_generator(word: str
                         ) -> List[str]:
        while True:
            samples = np.random.choice(
                vocab, neg_sample_size, p=noise_distribution
            )
            if word not in samples:
                return samples
            
    data['negative_samples'] = data['centre_word'].apply(samples_generator)
    return data
    

def sigmoid(x: np.ndarray       
            ) -> np.ndarray:
    """ 
    The important sigmoid function for squashing the values between 0 and 1.
    """
    return 1/(1+np.exp(-x))


def cost_derivation(update_param: np.ndarray,
                    dependent_param: np.ndarray,
                    label: int
                    ) -> np.ndarray:
    """ 
    Derivation of the cost function for updating the gradients.
    """
    return (sigmoid(update_param @ dependent_param) - label)*dependent_param 


def stochastic_gradient_descent(example: pd.Series,
                                matrices: Matrices,
                                word2onehot: Dict[str, int],
                                learn_rate: float
                                ) -> Matrices:
    """
    Stochastic Gradient Descent for training the word2vec model.
    """
    
    # Get hidden layer (projection vector), positive context word, and 
    # list of negative context words, all as vectors.
    proj_vector, c_positive, w_negative = (
        matrices.embedding[word2onehot[example['centre_word']]],
        matrices.context[word2onehot[example['context_word']]],
        [matrices.context[word2onehot[sample]]
         for sample in example['negative_samples']] 
    )
    
    # Obtain vectors for updating gradients. 
    embedding_gradient = (
        cost_derivation(proj_vector, c_positive, 1) + 
        sum(cost_derivation(proj_vector, c_negative, 0)
            for c_negative in w_negative)
    )
    pos_context_gradient = cost_derivation(c_positive, proj_vector, 1)
    neg_context_gradients = (
        (c_negative, cost_derivation(c_negative, proj_vector, 0))
        for c_negative in w_negative    
    )
    
    # Update parameters.
    proj_vector[:] = proj_vector - learn_rate*embedding_gradient 
    c_positive[:] = c_positive - learn_rate*pos_context_gradient
    for c_negative, neg_context_gradient in neg_context_gradients:
        c_negative[:] = c_negative - learn_rate*neg_context_gradient 
    
    return matrices


def train_w2v(skipgram_data: pd.DataFrame,
              matrices: Matrices,
              word2onehot: dict,
              vocabulary: np.ndarray,
              noise_distribution: List[int],
              negative_sample_size: int,
              num_epochs: int,
              learn_rate: float,
              verbose: bool
              ) -> Matrices:
    """ 
    Train the model.
    """
    
    start_time = time.time()
    for i in range(num_epochs):
        epoch_time = time.time()
        
        # First, apply some weighted randomised negative sampling to the data.
        skipgram_data = negative_sampling(
            skipgram_data, 
            vocabulary, 
            noise_distribution,
            negative_sample_size
        )
        
        # print(skipgram_data.head(15))
        for _, example in skipgram_data.iterrows():
            
            # Use stochastic gradient descent to update the parameters (i.e., 
            # the embedding and context matrices).
            matrices = stochastic_gradient_descent(
                example, matrices, word2onehot, learn_rate
            )
        
        if verbose:
            print(f"Epoch {i+1}: {time.time()-epoch_time} seconds.")

    if verbose:
        print(f"\n{num_epochs} epochs in {time.time()-start_time} seconds.\n")            
    return matrices


def word2vec_main(df: pd.DataFrame,
                  window_size: int = WINDOW_SIZE,
                  layer_size: int = VECTOR_DEPTH,
                  negative_sample_size: int = NEG_SAMPLE_SIZE,
                  num_epochs: int = NUM_EPOCHS, 
                  learn_rate: float = LEARN_RATE,
                  dist_alpha: float = DIST_ALPHA,
                  verbose: bool = VERBOSE
                  ) -> tuple:
    """ 
    Main function for the word2vec algorithm/procedure.
    """ 
    
    # Separate the keywords and the sentences from the DataFrame
    keywords, corpus = separate_keywords_corpus(df)
    
    # Preprocess the corpus
    print("Commencing data preprocessing...")
    w2vp = W2VPreprocessor(keywords, corpus)
    
    # Obtain a noise distribution containing vocabulary for use within the
    # negative sampling
    noise_distribution = get_noise_distribution(
        w2vp.preprocess_corpus, w2vp.vocabulary, dist_alpha
    )
    
    # Encode the words such that they correspond with matrix indices
    word2onehot = one_hot_vocab_encoding(w2vp)
    
    # Initialise the embedding and contex matrices
    matrices = get_matrices((w2vp.vocab_size, layer_size))

    # Initialise the dataset via the skipgrams methodology
    skipgram_data = obtain_skipgram_dataset(
        w2vp.preprocess_corpus, 
        window_size
    )
    # print(skipgram_data.head(15))
    
    if verbose:
        print(
            f"\nTraining Word2vec, via skipgrams and negative sampling, " 
            f"using the following parameters. \n"
            f"Vocabulary size:           {w2vp.vocab_size} \n"
            f"Window size:               {window_size} \n"
            f"Word vector depth:         {layer_size} \n"
            f"Negative sample size:      {negative_sample_size} \n"
            f"Distribution parameter:    {dist_alpha} \n"
            f"Number of epochs:          {num_epochs} \n"
            f"Learning rate (alpha):     {learn_rate} \n"
        )
    
    # Train the model to obtain the final embedding and context matrices. The
    # embedding matrix will contain the final word vectors, which can be 
    # extracted using the one hot encodings of the original vocabulary
    matrices = train_w2v(
        skipgram_data, 
        matrices, 
        word2onehot, 
        w2vp.vocabulary, 
        noise_distribution,
        negative_sample_size, 
        num_epochs,
        learn_rate,
        verbose
    )
        
    return word2onehot, w2vp, matrices


def cosine_sim(a: np.ndarray, 
                b: np.ndarray 
                ) -> float:
    """ 
    Function for calculating the cosine similarity of two input vectors, which
    represents the semantic similarity of the words they represent. This 
    specific implementation enforces outputs between 0 to 1, where greater
    values imply greater similarity.
    """
    return (
        1 + a.dot(b) / 
        (np.linalg.norm(a)*np.linalg.norm(b))
    ) / 2


def get_cosine_similarities(keywords: List[str],
                            matrices: Matrices,
                            word2onehot: Dict[str, int]
                            ) -> None:
    """ 
    Function for iterating through all combinations of the inputted keywords 
    and calculating (and printing) their cosine similarities.
    """
    for i in combinations(keywords, 2):
        print(i[0], i[1], cosine_sim(
            matrices.embedding[
                word2onehot[i[0]]], matrices.embedding[word2onehot[i[1]]
            ]
        ))


def get_semantic_dist_matrix(target_words: List[str],
                             word2onehot: Dict[str, int], 
                             matrices: Matrices
                             ) -> np.ndarray:
    """ 
    Function that uses the word embeddings obtained via the word2vec algorithm
    and creating a semantic distance matrix based on the cosine similarity 
    between the vectors of specific input words.
    """
    n = len(target_words)
    distance_matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i, n):
            vec1 = matrices.embedding[word2onehot[target_words[i]]]
            vec2 = matrices.embedding[word2onehot[target_words[j]]]
            distance = cosine_sim(
                vec1, vec2
            )
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


def matrix2dataframe(keywords: List[str],
                     matrix: np.ndarray,
                     path: str = DISTANCE_XLSX
                     ) -> pd.DataFrame:
    """ 
    Function for taking a numpy matrix and converting it into a 2D pandas 
    DataFrame, which is then converted to an xlsx file.
    """
    sem_dist_df = pd.DataFrame(
        data=matrix, index=keywords, columns=keywords
    )
    sem_dist_df.to_excel(path, index_label='Keywords')
    return sem_dist_df


def problem3() -> None:
    """ 
    Container function for the methods associated with problem 3.
    """
    try:
        print("Reading in the text files for each keyword...")
        kw_df = collect_keyword_dataframe(collect_all=COLLECT_ALL)  
    except FileNotFoundError:
        raise Exception("ERROR: The 'articles' folder was not found. Please "
                        "ensure that the functions for problems 1 and 2 have "
                        "been applied first.")
    
    word2onehot, w2vp, matrices = word2vec_main(kw_df)
    sem_dist_mat = get_semantic_dist_matrix(
        w2vp.target_words, word2onehot, matrices
    )
    
    try:
        matrix2dataframe(w2vp.keywords, sem_dist_mat)
    except PermissionError:
        print("ERROR: Unable to write 'distance.xlsx'. Please ensure that "
              "this file is not open.")
    
    # Functions for visualisations:
    reduce_and_plot_word_vectors(
        w2vp.keywords, [word2onehot[w] for w in w2vp.target_words], matrices
    )
    nearest_neighbours( 
        w2vp, word2onehot, matrices        
    )
    similarity_distributions(
        kw_df, w2vp, word2onehot, matrices
    )


## Problem 4
"""
The following section of code comprises methods for visualising/illustrating
various parts of my semantic distance algorithm. This entails visualisations 
for the differences within the corpus (collected from BBC news) before and 
after preprocessing, and visualisations of the semantic distances between the 
keywords, and other words (to sanity check the accuracy).

The functions here are dependent on the methods from the prior sections. That 
is, the articles for the keywords must be collected and stored as text files, 
and the semantic distances must be calculated (and stored in distance.xlsx). 

All of the visualisations are produced entirely using seaborn, with matplotlib
used only for mundane tasks such as setting indices and for displaying/saving
the visualisations. 
"""

from collections import Counter
from itertools import chain
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})


def read_in_distance_matrix(path: str = DISTANCE_XLSX
                            ) -> pd.DataFrame:
    """ 
    Read in the semantic distances matrix as a pandas DataFrame.
    """
    dist_matrix = pd.read_excel(path, index_col=0)
    return dist_matrix


def reduce_and_plot_word_vectors(keywords: List[str],
                                 target_words: List[str],
                                 matrices: Matrices
                                 ) -> None:
    """ 
    Decompose the multidimensional vectors into 2D ones, so that they can be 
    plotted on a relation plot. This process uses PCA for the dimensionality 
    reduction. Then, standardise the coordinates using the sigmoid function. 
    The keywords are then plotted via a seaborn relplot, with the maximum and 
    minimum x and y values set to 1 and 0 respectively, as these are the 
    maximum/minimum values the coordinates can take (after being processed with 
    the sigmoid function). 
    Note that the word2vec parameters are sensitive, and slight changes can 
    lead to very different plots, location wise. The distances, however, will
    generally remain consistent (assuming sufficient epochs).
    """
    
    # Decompose the word vectors via Principle Component Decomposition such 
    # that they now have 2 dimensions and can be more easily visualised
    pca = PCA(n_components=2, random_state=1)
    decomposed = pca.fit_transform(matrices.embedding)
    decomposed = np.array([sigmoid(x) for x in decomposed])
    coords = np.array([decomposed[w] for w in target_words])
    
    # Create a dataframe for the coordinates of the 2D vectors
    coords_df = pd.DataFrame([
        {"keyword": keyword, "x-axis": coords[i, 0], "y-axis": coords[i, 1]}
         for i, keyword in enumerate(keywords)    
    ])
    
    # Plot the 2D vectors with seaborn relplot
    sns.relplot(
        data=coords_df, 
        x="x-axis", 
        y="y-axis", 
        hue="keyword"
    )

    sns.utils.plt.show()


def get_similar(target_word: str, 
                word2onehot: Dict[str, int], 
                matrices: Matrices, 
                vocabulary: np.ndarray,
                n_similar: int = 10,
                sort: bool = False,
                ) -> list:
    """
    Function to obtain a list of tuples, containing words and the cosine 
    similarity of these words with the input target word.
    """
    target_word_emb = matrices.embedding[word2onehot[target_word]]
    values = []
    for word in vocabulary:
        if word != target_word:
            word_emb = matrices.embedding[word2onehot[word]]
            values.append((word, cosine_sim(target_word_emb, word_emb)))

    if not sort:
        return [(w,v, target_word) for w, v in sorted(
            values, key = lambda x: x[1], reverse = True
        )][:n_similar]
    else:
        return values


def nearest_neighbours(w2vp: W2VPreprocessor,
                       word2onehot: Dict[str, int],
                       matrices: Matrices,
                       n_nearest: int = 3
                       ) -> None:
    """ 
    In order to verify the success of the model more generally, visualise the
    nearest semantic neighbours to the inputted keywords by creating bar charts
    corresponding to the similarity proportions of the k-nearest neighbours.
    """
    cols = ["b", "g"]
    f, ax = sns.utils.plt.subplots(figsize=(11, 8))
    
    # Collect the nearest neighbours to the keywords and store this info. in a
    # DataFrame
    neighbours = list(chain(*[
        get_similar(
            word, word2onehot, matrices, w2vp.vocabulary, n_similar=n_nearest
        ) for word in w2vp.target_words
    ]))
    
    dfs = []
    for i in range(n_nearest-1, -1, -1):
        new_df = pd.DataFrame([
            {'Keyword': w2vp.target2original[neighbours[j][2]],
             'Neighbour': neighbours[j][0],
             'Similarity': (neighbours[j][1] + sum(
                     neighbours[j-c-1][1] for c in range(i)
                ))}
            for j in range(i, len(neighbours), n_nearest)
        ])
        dfs.append(new_df)    
    for i, df in enumerate(dfs):
        
        # Visualise the keyword neighbours via a barplot
        bp = sns.barplot(
            data=df,
            y="Keyword", x="Similarity", 
            orient="h", color=cols[i%2]
        )
        
        # label the bars
        y_locs = bp.get_yticks()
        for idx, row in df.iterrows():
            bp.text(x=row["Similarity"], y=y_locs[idx], s=row["Neighbour"],
                    color="black", ha="right")

    sns.despine(left=True, bottom=True)
    sns.utils.plt.show()   
    
    
def similarity_distributions(kw_df: pd.DataFrame,
                             w2vp: W2VPreprocessor,
                             word2onehot: Dict[str, int],
                             matrices: Matrices
                             ) -> None:
    """
    Visualise the distribution of similarity scores for each of the keywords, 
    and use this to analyse the effect of data size pertaining to keywords and
    accuracy of similarity scores. Visualisation achieved using a violinplot.
    """
    
    # Create a dictionary containing the value counts of each keyword in the 
    # initial articles
    value_counts = {
        w2vp.original2target[word]: len(kw_df[kw_df["keyword"] == word]) 
        for word in kw_df["keyword"].unique()
    }

    # Collect the similarities of each keyword
    to_df = []
    for word in w2vp.target_words:
        to_df += [
            {'Keyword': f"{w2vp.target2original[word]}\n"
                        f"(Sentence count={value_counts[word]})", 
             'count': value_counts[word],
             'Similarity Value': neighbour[1]}
            for neighbour in get_similar(
                word, word2onehot, matrices, w2vp.vocabulary, sort=True
            )
        ]
        
    # Visualise the similarity distributions via a violineplot
    similarity_df = pd.DataFrame(to_df)
    similarity_df.sort_values(by="count", inplace=True)
    
    v_plot = sns.violinplot(
        data=similarity_df, x='Keyword', y='Similarity Value', 
        order=similarity_df["Keyword"].unique()
    )
    
    # Rotate the x-axis labels such that they are visible (i.e., they don't 
    # overlap with one another)
    v_plot.set_xticklabels(
        labels=v_plot.get_xticklabels(), 
        rotation=55, 
        horizontalalignment='right',
        fontweight='light'
    )
    
    sns.utils.plt.show()


def keywords_countplot(data: pd.DataFrame
                       ) -> None:
    """
    Illustrate the distribution of sentences per keyword in the data via a 
    seaborn countplot, and label each bar with the precise counts
    """
    ordered_keywords = data["keyword"].value_counts().keys() 
    
    # Initialise the countplot, using the keywords as the x-axis and their 
    # respective counts as the y-axis
    c_plt = sns.countplot(
        x="keyword",             
        palette=sns.color_palette("crest"),
        data=data,                
        order=ordered_keywords    
    )
    
    # Rotate the x-axis labels such that they are visible (i.e., they don't 
    # overlap with one another)
    c_plt.set_xticklabels(
        labels=c_plt.get_xticklabels(), 
        rotation=55, 
        horizontalalignment='right',
        fontweight='light'
    )
    
    # Incoprorate annotation so the specific values of each count is shown
    for c, label in zip(c_plt.patches, data["keyword"].value_counts()):
        c_plt.annotate(label, (c.get_x()+0.25, c.get_height()+0.5))
        
    sns.utils.plt.show()
    
    
def displot_all_sentence_lengths(data: pd.DataFrame 
                                 ) -> None:
    """ 
    Create a distribution plot to visualise the distribution of sentence 
    lengths present in the collected data. This is done by first creating a new 
    column in the dataframe by applying a len function to each value in the 
    'sentence' column and then using seaborn's displot method
    """
    data['sentence_length'] = data['sentence'].apply(
        lambda s: len(s.split(' '))
    )
    dp = sns.displot(data=data['sentence_length'], kde=True)
    
    # Set lower limit to zero on x-axis
    _, upper = dp.ax.get_xlim()
    dp.ax.set_xlim(0, upper)
    sns.utils.plt.show()


def displot_sentence_lengths_per_keyword(data: pd.DataFrame 
                                 ) -> None:
    """ 
    Akin to the previous function, except this one will generate multiple 
    curves, one for each keyword, allowing us to visualise the distribution of
    sentence lengths per keyword. As a bonus, it also illustrates how the 
    distribution of sentence lengths changes as the number of sentences is
    increased.
    """
    data['sentence_length'] = data['sentence'].apply(
        lambda s: len(s.split(' '))
    )
    sns.displot(
        data=data, 
        x="sentence_length", 
        hue="keyword", 
        kind="kde",
    )

    sns.utils.plt.show()


def visualise_initial_most_frequent_words(data: pd.DataFrame 
                                          ) -> None:
    """ 
    Visualise the most frequent words in the entire dataset prior to applying
    any data cleaning methods (such as stop word removal)
    """
    entire_corpus = " ".join(
        [sentence.lower() for sentence in data["sentence"]]
    ).split()
    
    # Remove stopwords
    entire_corpus = list(filter(lambda x: x not in STOPWORDS, entire_corpus))
    
    # Obtain the most frequent words
    corpus_counter = Counter(entire_corpus)
    most_freq = corpus_counter.most_common(50)

    # Create a dataframe for these words
    to_df = [{'word': word[0], 'count': word[1]} for word in most_freq]
    word_freq_df = pd.DataFrame(to_df)
    
    # Display barplot 
    sns.barplot(
        data=word_freq_df, y='word', x='count', 
        palette=sns.color_palette("crest")
    )
    sns.utils.plt.show()


def visualise_preprocessed_most_frequent_words(corpus: List[str]
                                               ) -> None:
    """ 
    Similar method to previous, except this time illustrate the word 
    frequencies in the corpus after data cleaning has been applied.
    """

    corpus_words = [word for sentence in corpus for word in sentence.split()]
    
    # Obtain the most frequent words
    corpus_counter = Counter(corpus_words)
    most_freq = corpus_counter.most_common(50)

    # Create a dataframe for these words
    to_df = [{'word': word[0], 'count': word[1]} for word in most_freq]
    word_freq_df = pd.DataFrame(to_df)
    
    # Display barplot 
    sns.barplot(
        data=word_freq_df, y='word', x='count', 
        palette=sns.color_palette("crest")
    )
    sns.utils.plt.show()
    

def visualise_distances(distances: pd.DataFrame
                        ) -> None:
    """
    Construct a heatmap using seaborn to visualise the semantic distance matrix
    containing each of the keywords.
    Specific colour palette chosen as it has a more clear hierarchy
    """
    
    # Initialise the heatmap
    hm = sns.heatmap(
        data=distances,
        annot=True,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette("crest", as_cmap=True)
    )
    
    # Handle the cut-off top and bottom that occasionally occurs
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    
    # Save the heatmap to a file
    # hm.savefig("distance_heatmap.png")
    sns.utils.plt.show()
    

def problem4() -> None:
    """ 
    Container function for the methods for problem 4.
    """
    
    # Obtain the dataframe containing lines from the articles and the keyword
    # with which the article is associated.
    try:
        kw_df = collect_keyword_dataframe(collect_all=COLLECT_ALL)  
    except FileNotFoundError:
        raise Exception("[ERROR] The 'articles' folder was not found. Please "
                        "ensure that the functions for problems 1 and 2 have "
                        "been applied first.")
    
    keywords, corpus = separate_keywords_corpus(kw_df)
    
    # Complete the preprocessing for the corpus for comparison's sake
    w2vp = W2VPreprocessor(keywords, corpus)
    
    # Display the counts of sentences collected associated with each keyword
    keywords_countplot(kw_df)
    
    # Create displots for the lengths of sentences in the corpus
    displot_all_sentence_lengths(kw_df)
    displot_sentence_lengths_per_keyword(kw_df)
    
    # Visualise the semantic distances
    try:
        distances = read_in_distance_matrix()
    except FileNotFoundError:
        raise Exception("[ERROR] The distance matrix excel file was could not "
                        "be found! Please ensure the methods for the "
                        "preceding problems have been run first.")
    visualise_distances(distances)
    
    # Visualise word frequency before and after preprocessing
    visualise_initial_most_frequent_words(kw_df)
    visualise_preprocessed_most_frequent_words(w2vp.corpus)


def main() -> None:
    """ 
    Main function containing the container functions for each problem. These 
    may be commented in and out in order to only run the functions for specific
    problems - but keep in mind that some problems are not independent. 
    """
    print("Beginning program...")
    t1 = time.time()
    
    print("Commencing functions for problem 1.")
    web_pages = problem1()
    t2 = time.time()
    print(f"Problem 1 elapsed time: {t2 - t1 :.3f} seconds")

    print("Commencing functions for problem 2.")
    problem2(web_pages) 
    t3 = time.time() 
    print(f"Problem 2 elapsed time: {t3 - t2 :.3f} seconds") 

    print("Commmencing functions for problem 3.")
    problem3()
    t4 = time.time()
    print(f"Problem 3 elapsed time: {t4 - t3 :.3f} seconds")

    print("Comencing functions for problem 4.")
    problem4()
    t5 = time.time()
    print(f"Problem 4 elapsed time: {t5 - t4 :.3f} seconds")
    
    print(f"Total elapsed time: {t5 - t1 :.3f} seconds")
    

if __name__ == '__main__':
    main()

"""
Note on run time:
    Generally, the program will take between 10 and 15 minutes to complete 
    entirely - a full breakdown can be found in the report.
"""
