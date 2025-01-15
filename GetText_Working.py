from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from joblib import load
import warnings

warnings.filterwarnings("ignore")

presence_classifier = load('presence_classifier.joblib')
presence_vect = load('presence_vectorizer.joblib')
category_classifier = load('category_classifier.joblib')
category_vect = load('category_vectorizer.joblib')

def get_all_text_from_site(url):

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')

    # Set up the Chrome WebDriver without additional options
    driver = webdriver.Chrome(options=options)

    # Open the URL
    driver.get(url)

    # Wait for the page to load
    time.sleep(5)  # Adjust this delay as necessary to ensure the page loads completely

    # Extract all text elements from the page and split by newline
    elements = driver.find_elements(By.XPATH, '//*')
    all_text = []
    for element in elements:
        texts = element.text.split('\n')
        for text in texts:
            if text.strip():  # This checks if the text is not just whitespace
                all_text.append(text.strip())

    # Close the browser
    driver.quit()

    return all_text

# Example usage
url = 'https://www.damilano.com/collections/women-handbags/products/medium-wax-quilting-leather-satchel-misa-rose-lb-00309amisa-rosewax-quilting'
all_text = get_all_text_from_site(url)
#print("\n".join(all_text))
#print(all_text)

Darkpatterns=[] # Consists dark pattern strings
for token in all_text:
    result = presence_classifier.predict(presence_vect.transform([token]))
    if result=="Dark":
        Darkpatterns.append(token)

print("Dark patter Strings:\n",Darkpatterns)

len_darkpatterns= len(Darkpatterns)

DarkpatternsClassification=[]
unique_pattern_list=[]
for tokenDark in Darkpatterns:
    res = category_classifier.predict(category_vect.transform([tokenDark]))
    if res[0] not in unique_pattern_list:
        unique_pattern_list.append(res[0])
    DarkpatternsClassification.append([tokenDark,res[0]])

def Dark_Score(n):
    score=100
    for pattern in n:
        if pattern=='Scarcity' or pattern=='Urgency' or pattern=='Misdirection':
            score-=5
        elif pattern=='Obstruction'or pattern=='Sneaking':
            score-=15
        else:
            score-=20
    return(score)

print("\nClassification\n",DarkpatternsClassification)
print("\nFinal Outcome: ")
print("Total Dark Patterns:",len_darkpatterns)
print("All the unique dark patterns:",unique_pattern_list)
print("Score",Dark_Score(unique_pattern_list))
import databaseDP as ab
ab.upload_data(url, DarkpatternsClassification, len_darkpatterns, unique_pattern_list, Dark_Score(unique_pattern_list))
