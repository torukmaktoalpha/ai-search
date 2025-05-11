from flask import Flask, request, render_template
import requests


from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from googleapiclient.discovery import build  # Google API client

app = Flask(__name__)

# Google Custom Search API credentials
API_KEY = ""  # Replace with your API key
CSE_ID = ""  # Replace with your search engine ID

def search_web(query):
    # Initialize the Google search API client
    service = build("customsearch", "v1", developerKey=API_KEY)
    
    # Perform the search, limiting to 2 results
    res = service.cse().list(q=query, cx=CSE_ID, num=2).execute()  # Limiting to 2 results
    results = []
    
    # Extract URLs from search results
    if 'items' in res:
        for item in res['items']:
            results.append(item['link'])

    # Debug: Print retrieved URLs
    print(f"Retrieved URLs from Google Search API: {results}")
    
    return results

def scrape_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        
        # Debug: Print the scraped content length
        print(f"Scraped content from {url} (Length: {len(content)} characters)")
        
        return content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def summarize_text_bart(text):
    # Load the BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    
    # Generate summary using the model
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary to readable text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        urls = search_web(query)

        full_text = []
        for url in urls:
            content = scrape_content(url)
            if content:  # Only append non-empty content
                full_text.append(content)

        if full_text:
            combined_text = ' '.join(full_text)
            summary = summarize_text_bart(combined_text)  # Use BART for summarization
            return render_template('result.html', summary=summary)
        else:
            # Debug: Print if no content was found
            print("No content scraped from any URL.")
            return render_template('result.html', summary="No content found.")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
