import email
import html2text
import imaplib
import openai
import os
import pandas as pd
import re
import streamlit as st

from bs4 import BeautifulSoup as BS
from datetime import datetime, timedelta
from scipy import spatial
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
from sentence_transformers import SentenceTransformer

# Secrets
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
EMAIL = os.environ['EMAIL']
PASSWORD = os.environ['PASSWORD']
IMAP_SERVER = os.environ['IMAP_SERVER']

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Set device for sentence transformer inference
device = "cpu"

# GeckoDriver location
driver_path = "./assets/firefox"

# URL of HAD site
HAD_URL = 'https://www.had.de/onlinesuche_einfach.html'


def load_sentence_transformer():
    return SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer', device=device)


def get_mail_data(subfolder, days):
    """Fetch emails from dedicated email account

    Arguments:
    subfolder: str -- name of subfolder in email account
    days: int -- download emails of the past 'days' days

    Returns:
    emails: List -- email metadata and contents
    """

    html_converter = html2text.HTML2Text()
    html_converter.ignore_links = True

    imap_server = imaplib.IMAP4_SSL(IMAP_SERVER)
    imap_server.login(EMAIL, PASSWORD)
    imap_server.select(f'inbox/{subfolder}')

    start_date = (datetime.now() - timedelta(days=days)).strftime('%d-%b-%Y')
    criterion = f"(SINCE {start_date})"

    _, data = imap_server.search('UTF-8', criterion)
    if not data:
        return []

    emails = {"total": None, "data": []}

    n_mails = 0
    for num in data[0].split():
        n_mails += 1
        email_content = ""
        _, msg_data = imap_server.fetch(num, '(RFC822)')
        email_msg = email.message_from_bytes(msg_data[0][1])
        if email_msg.get_content_type() == "text/html":
            payload_str = email_msg.get_payload(decode=True).decode('utf-8')
            email_content = html_converter.handle(payload_str)
        elif email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == 'text/plain':
                    charset = part.get_content_charset()
                    if charset is None:
                        email_content += part.get_payload()
                    else:
                        email_content += part.get_payload(decode=True).decode(charset)
                elif part.get_content_type() == 'text/html':
                    payload_str = part.get_payload(decode=True).decode('utf-8')
                    email_content += payload_str
        else:
            email_content = email_msg.get_payload()

        emails['data'].append({'subject': email_msg['Subject'],
                               'from': email_msg['From'],
                               'date': email_msg['Date'],
                               'content': email_content})

    emails["total"] = n_mails

    return emails


def get_had_table(days):
    """Scrape table with open calls from HAD website

    Arguments:
    None

    Returns:
    had_table: Pandas dataframe -- HAD table with open calls
    """

    had_table_filtered = pd.DataFrame()

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Remote(
        command_executor='http://localhost:4444/wd/hub',
        desired_capabilities=DesiredCapabilities.FIREFOX,
        options=options
    )
    driver.get(HAD_URL)

    select_element = Select(driver.find_element(By.NAME, "L_CAT"))
    select_element.select_by_value("SQLB")

    radio_button = driver.find_element(By.XPATH, '//input[@type="radio" and @value="500"]')
    radio_button.click()

    submit = driver.find_element(By.CLASS_NAME, "submit")
    submit.click()

    wait = WebDriverWait(driver, 10)

    try:
        content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        table_html = content.get_attribute('outerHTML')
        pattern = r'<div class="small">.*?</div>'  # because it scrambles table
        table_clean = re.sub(pattern, '', table_html)
        table_soup = BS(table_html, 'lxml')
        table_rows = table_soup.find_all('tr')
        table_forms = [str(row.find('form')) for row in table_rows if row.find('form')]

        had_table = pd.read_html(table_clean, encoding='utf-8', header=0)[0]
        had_table.dropna(axis=0, how='all', inplace=True)
        had_table.drop(had_table.columns[[0]], axis=1, inplace=True)
        had_table.drop(had_table.columns[[-1]], axis=1, inplace=True)
        had_table.reset_index(drop=True, inplace=True)

        had_table.rename(columns={'VerfahrenLeistung': 'call'}, inplace=True)
        had_table.rename(columns={'veröffentlicht am/ Ablauftermin': 'date'}, inplace=True)
        had_table.rename(columns={'Vergabestelle/Ort': 'client_info'}, inplace=True)
        had_table.rename(columns={'Ort der Leistung': 'place'}, inplace=True)
        had_table['date'] = had_table['date'].str.split().str[0]
        had_table['date'] = pd.to_datetime(had_table['date'], format='%d.%m.%Y')

        if len(table_forms) == len(had_table):
            had_table["link"] = table_forms
        else:
            raise ValueError("Number of form elements doesn't match the number of rows in the dataframe")

        timeframe = datetime.now() - timedelta(days=days)
        had_table_filtered = had_table[had_table['date'] >= timeframe]

    except TimeoutException:
        st.warning("Time out: Content konnte nicht geladen werden")
        driver.quit()

    driver.quit()

    return had_table_filtered


def preprocess_nl(content, abo):
    """Preprocess newsletter for downstream evaluation

    Keyword arguments:
    content: str --> text of email body
    abo: str --> newsletter source (currently only: service.bunde.de or Bundesanzeiger)

    Returns:
    call_list: list[Optional[list,str]] --> depending on abo
    df: Pandas dataframe --> one row for each call in newsletter
    """

    df = pd.DataFrame()
    call_list = []
    if abo == "service.bund.de":
        text_pp = []
        lines = content.splitlines()
        for k, line in enumerate(lines):
            if line[:4] == "http":
                line_count = 0
                for ll in range(k + 1, k + 10):
                    line_count += 1
                    if lines[ll] == '':
                        text_pp.append(lines[k - 1:k + line_count])
                        break
        call_list = [[item[0], item[1], item[2:]] for item in text_pp]
        df = pd.DataFrame(call_list, columns=['call', 'url', 'client_info'])

    elif abo == "Bundesanzeiger":
        call_list = []
        soup = BS(content, 'html.parser')
        table = soup.find('table')
        df = pd.read_html(str(table), encoding='utf-8', header=0)[0]
        df['url'] = [tag.get('href') if tag.has_attr('href') else "kein Link" for tag in table.find_all('a')]
        df['Titel'] = df.Titel.apply(lambda x: x.replace("» ", ""))
        df.rename(columns={'Titel': 'call'}, inplace=True)
        for j, row in df.iterrows():
            call_string = f"{abo}: Ausschreibung {str(j)}<br>{row.call}<br>{row.url}<br>"
            call_list.append(call_string)

    return df, call_list


def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    """Calculate distances between query and database vectors

    Keyword arguments:
    query_embedding: array --> embedding vector for dept. name
    embeddings: list --> list of embedding vectors for call titles
    distance_metric: str --> defaults to cosine similarity

    Returns:
    distances: list --> cosine similarity for each pair of dept. name and call title embeddings
    """

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    distances = [1. - distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]

    return distances


def encode_calls(df, bi_encoder):
    """Encode the call titles using the Bi-Encoder and store as new df column

        Keyword arguments:
        df: Pandas dataframe --> call data (from preprocess_nl)
    """

    df['embeddings'] = df.call.apply(
        lambda x: bi_encoder.encode(x, show_progress_bar=False, device=device))


def evaluate_calls(df, k, query, bi_encoder):
    """Evaluate calls newsletter by predicting semantic similarity between dept. query and call titles

    Keyword arguments:
    df: Pandas dataframe --> call data (from preprocess_nl)
    k: int --> Top-k, keep the k calls with the highest cosine similarity to query
    query: str --> query as defined in departments.json
    encoder: class --> sentence transformer model for feature extraction

    Returns:
    df_top_k: Pandas dataframe --> call data for k calls with highest cosine similarity
    """

    # Encode the query using the Bi-Encoder
    query_embedding = bi_encoder.encode(query, show_progress_bar=False, device=device)

    # Calculate similarity between query embedding and the embeddings of all call titles and store as new df column
    df['be_scores'] = distances_from_embeddings(query_embedding, df['embeddings'].values, distance_metric='cosine')

    # Create copy of dataframe with the k rows that have the highest be_scores
    top_k_be_scores = df.nlargest(k, 'be_scores')['be_scores']
    df_top_k = df[df['be_scores'].isin(top_k_be_scores)].copy()

    return df_top_k


def ask_llm(model, d_description, call_title):
    """Check top_k calls as determined by evaluate_calls by LLM

    Keyword arguments:
    d_name: str --> name of research dept.
    call_title: str --> title of call to evaluate

    Returns:
    evaluation: str --> model answer
    """

    # Construct system prompt
    delimiter = "####"
    system_prompt = f"""
        Du bekommst den Titel einer öffentlichen Ausschreibung.
        Der Titel der Ausschreibung wird durch {delimiter} begrenzt.

        Deine Aufgabe ist es, zu prüfen, ob sich die Forschungsabteilung auf die Ausschreibung bewerben könnte.
        Antworte mit einem Score auf einer Skala von 1 bis 5, wobei 1 'auf keinen Fall bewerben' und 5 'unbedingt bewerben' bedeutet.
        Nenne in Deiner Antwort nur den Score. Verzichte auf jede sonstige Äußerung.

        Forschungsabteilung: {d_description}

        Denk daran: Antworte nur mit dem Score! 
        """

    # Construct  model prompt
    model_prompt = delimiter + call_title + delimiter

    # Define the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {'role': "user", "content": model_prompt}
    ]

    # Generate answer
    evaluation = ""
    try:
        evaluation = openai.ChatCompletion.create(
            model=model,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            messages=messages
        )["choices"][0]["message"]["content"]
    except openai.error.APIError as e:
        st.warning(f"Language model API returned an API Error: {e}")
    except openai.error.APIConnectionError as e:
        st.warning(f"Failed to connect to language model API: {e}")
    except openai.error.RateLimitError as e:
        st.warning(f"Language model API request exceeded rate limit: {e}")
    except openai.error.InvalidRequestError as e:
        st.warning(f"Language model API reported invalid request error: {e}")
    except openai.error.ServiceUnavailableError as e:
        st.warning(f"Language model API unavailable: {e}")

    return evaluation


def keyword_check(title):
    """Filter out call titles, which contain pre-defined keywords

    Keyword arguments:
    title: str --> call title

    Returns:
    keyword_flag: bool --> False, if title contains at least one keyword, else True
    """

    with open("./assets/negative_keywords.txt", "r", encoding="utf-8") as file:
        negative_keywords = file.read().lower().split("\n")

    keyword_flag = True
    for keyword in negative_keywords:
        if keyword in title.lower():
            keyword_flag = False

    return keyword_flag


def format_call(call):
    """Format call for pretty display

    Keyword arguments:
    call: list --> list with call data (from preprocess_nl)

    Returns:
    formatted_call: str --> call as string for nice HTML display
    """

    formatted_call = call[0] + "<br>" + call[1] + "<br>"
    for line in call[2]:
        formatted_call += line + "<br>"

    return formatted_call


def format_had_link(link):
    link = link.replace("<table>", "")
    link = link.replace("</table>", "")
    link = link.replace("<tbody>", "")
    link = link.replace("</tbody>", "")
    link = link.replace("<tr>", "")
    link = link.replace("</tr>", "")
    link = link.replace("<td>", "")
    link = link.replace("</td>", "")

    return link
