from itertools import count

from nltk.corpus import stopwords
from requests import options
from selenium.webdriver import ActionChains
from tensorflow.keras.activations import softmax
import streamlit as st
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import time
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from textblob import Word

chrome_options = Options()
chrome_options.add_argument('--headless')  # Uncomment if you want to run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--window-size=1920,1080')
# ________________________________________________________________________________________________________________________

# important downloads
import nltk

nltk.download('omw-1.4')  # For multilingual WordNet support
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# ________________________________________________________________________________________________________________________

stop_words = stopwords.words('english')

model_file_path = '/Users/rishabhtiwari/Downloads/my_sentiment_model.pkl'
tokenizer_file_path = '/Users/rishabhtiwari/Downloads/tokenizer.pkl'


# ________________________________________________________________________________________________________________________

with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# ________________________________________________________________________________________________________________________

# Initialize session state for navigation and inputs

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "username" not in st.session_state:
    st.session_state.username = ""
if "password" not in st.session_state:
    st.session_state.password = ""

# Initialize session state
if "urls" not in st.session_state:
    st.session_state.urls = []

if "rankings" not in st.session_state:
    st.session_state.rankings = []

if "credentials_ok" not in st.session_state:
    st.session_state.credentials_ok = False

if "global_state_urls" not in st.session_state:
    st.session_state.global_state_urls = []

if "compute_signal" not in st.session_state:
    st.session_state.compute_signal = False

# ________________________________________________________________________________________________________________________

def amazon_login(driver, username, password):
    """
    Logs into Amazon using the provided credentials and returns a logged-in WebDriver instance.

    Args:
        username (str): Amazon username or email.
        password (str): Amazon password.

    Returns:
        WebDriver: Selenium WebDriver instance logged into Amazon.
        str: Message indicating success or failure.
    """
    try:
        # Initialize the Selenium WebDriver (adjust driver initialization as needed)

        # Click on the sign-in link
        # sign_in_link = WebDriverWait(driver, 10).until(
        #     EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        # )
        # sign_in_link.click()

        # Enter username
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_email"))
        )
        username_field.send_keys(username)
        username_field.submit()

        # Enter password
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        password_field.send_keys(password)
        password_field.submit()

        # Verify successful login
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )

        # Check if the text inside the account link has changed (indicating login success)
        if account_link.text != "Hello, sign in":
            return driver, "Login Successful"
        else:
            return None, "Login Failed"

    except Exception as e:
        print(f"Error during login: {e}")
        return None, "Login Failed"

# ________________________________________________________________________________________________________________________
def amazon_logout(driver):
    """
    Logs out from Amazon using the provided logged-in WebDriver instance.

    Args:
        driver (WebDriver): A Selenium WebDriver instance logged into Amazon.

    Returns:
        str: Message indicating success or failure.
    """
    try:
        # Hover over the account link to reveal the dropdown menu
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        actions = ActionChains(driver)
        actions.move_to_element(account_link).perform()

        # Wait for the sign-out button to appear and be clickable
        sign_out_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "nav-item-signout"))
        )
        sign_out_button.click()

        # Verify logout by checking if "Hello, sign in" text appears
        sign_in_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        if sign_in_link.text == "Hello, sign in":
            return "Logout Successful"
        else:
            return "Logout Failed"

    except Exception as e:
        print(f"Error during logout: {e}")
        return "Logout Failed"

# ________________________________________________________________________________________________________________________

def check_amazon_credentials(username, password):
    try:
        # Navigate to Amazon login page
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.amazon.com/")

        sign_in_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )

        sign_in_link.click()

        # Find and enter username
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_email"))
        )
        username_field.send_keys(username)
        username_field.submit()

        # Find and enter password
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        password_field.send_keys(password)
        password_field.submit()

        # Wait for the account link to appear
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )

        # Check if the text inside the account link is not "Hello, sign in"
        if account_link.text != "Hello, sign in":
            # Login successful, proceed to logout

            # Hover over the account link
            actions = ActionChains(driver)
            actions.move_to_element(account_link).perform()

            # Wait for the dropdown to appear
            dropdown = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "nav-a"))  # Adjust locator as needed
            )

            # Wait for the sign-out button to be clickable
            sign_out_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#nav-item-signout > span"))
            )

            # Click the sign-out button
            sign_out_button.click()

            # Confirm logout (if required)
            try:
                confirm_logout_button = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "auth-action-sign-out-form-submit"))
                )
                confirm_logout_button.click()
            except:
                pass  # No confirmation step required

            return "Login Successful"

        # If not logged in, check for specific error messages
        try:
            # Example: Check for a common error message
            error_message = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "auth-error-message-box"))
            ).text
            if "Invalid username or password" in error_message:
                return "Invalid Username or Password"
            else:
                return "Login Failed"
        except:
            return "Login Failed"

    except Exception as e:
        print(f"Error during login or logout: {e}")
        return "Login Failed"

# ________________________________________________________________________________________________________________________

def scrape_reviews(url, num):
    product_details = {"product_name": "", "reviews" : [], "stars" : []}
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)
    st.write(f'Progress of URL {num}')
    progress_bar = st.progress(0)  # Initialize the progress bar
    try:
        driver.get(url)
        product_title_element = driver.find_element(By.ID, "productTitle")
        if product_title_element is not None:
            product_details['product_name'] = product_title_element.text

        # Navigate to "See All Reviews"
        reviews_link = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@data-hook="see-all-reviews-link-foot"]')))
        reviews_link.click()

        amazon_login(driver, st.session_state.username, st.session_state.password)

        page = 1
        total_pages = 10  # Set this based on the number of pages you expect, or calculate dynamically


        while True:
            html_data = BeautifulSoup(driver.page_source, 'html.parser')
            is_next_page = html_data.find('li', {'class': 'a-last'})  # Find the "Next Page" button

            # Check if the "Next Page" button is disabled or does not exist
            if is_next_page is None or 'a-disabled' in is_next_page.get('class', ''):
                break  # Exit the loop if there's no next page or the button is disabled

            review_divs = html_data.find_all('div', {'id': re.compile(r'^customer_review-')})

            # Extract reviews, ratings, and metadata
            for review in review_divs:
                review_body = review.find('span', {'data-hook': 'review-body'})
                if review_body:
                    product_details['reviews'].append(review_body.get_text(strip=True))
                rating_element = review.find('i', {'data-hook': 'review-star-rating'})
                if rating_element:
                    rating_text = rating_element.find('span', {'class': 'a-icon-alt'}).text
                    product_details['stars'].append(rating_text.split()[0])

            # Check for next page
            next_page = html_data.find('li', {'class': 'a-last'})
            if not next_page or 'a-disabled' in next_page.get('class', ''):
                break

            next_page_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//ul[@class='a-pagination']//li[@class='a-last']/a")))
            next_page_button.click()
            wait.until(EC.staleness_of(next_page_button))
            page += 1

            # Update progress bar
            progress_bar.progress(page / total_pages)  # Update the progress bar

        amazon_logout(driver)
        driver.quit()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        progress_bar.progress(1)
        driver.quit()

    return product_details

# ________________________________________________________________________________________________________________________

def modifyReviews(reviews_array):
    """
    Preprocesses an array of input reviews to match the format used during training.

    Args:
        reviews_array (list): A list of reviews to be preprocessed.

    Returns:
        numpy.ndarray: The preprocessed reviews as a padded sequence.
    """
    processed_reviews = []
    for review in reviews_array:
        # Convert to lowercase and split
        review = ' '.join(review.lower() for review in review.split())

        # Replacing special characters using re.sub()
        review = re.sub(r"[^a-zA-Z\s]", '', review)  # Keep only letters and spaces

        # Replacing digits/numbers using re.sub()
        review = re.sub(r'\d+', '', review)  # Remove all digits

        # Removing emojis using re.sub()
        review = re.sub(r'[^\x00-\x7F]+', '', review)  # Remove non-ASCII characters (including emojis)

        # Removing stop words
        review = ' '.join(review for review in review.split() if review not in stop_words)

        # Lemmatization
        review = ' '.join([Word(word).lemmatize() for word in review.split()])

        processed_reviews.append(review)

    # Convert to sequence and pad
    X = tokenizer.texts_to_sequences(processed_reviews)
    X = pad_sequences(X, maxlen=103)

    return X
# ________________________________________________________________________________________________________________________

def process_url(url):
    # Scrape reviews from the URL
    reviews = scrape_reviews(url)
    print(reviews)
    # Process each review with the sentiment analysis model
    positive_scores = []
    for review in reviews:
        tokenized_review = tokenizer.texts_to_sequences([review])
        padded_review = pad_sequences(tokenized_review, maxlen=103)
        prediction = model.predict(padded_review)
        positive_scores.append(prediction[0][1])  # Assuming [negative, positive] output

    # Calculate average sentiment score
    avg_sentiment = sum(positive_scores) / len(positive_scores) if positive_scores else 0

    # Select top 5 reviews based on sentiment
    top_reviews = [review for _, review in sorted(zip(positive_scores, reviews), reverse=True)[:5]]

    return len(reviews), top_reviews, avg_sentiment

# ________________________________________________________________________________________________________________________

# Streamlit UI Configuration
st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon=":chart_with_upwards_trend:",
)

st.header("AI PICKER")


# Navigation buttons with consistent width
if st.sidebar.button("Home", key="home"):
    st.session_state.page = "Home"
if st.sidebar.button("Credentials", key="credentials"):
    st.session_state.page = "Credentials"
if st.sidebar.button("Page 2", key="page2"):
    st.session_state.page = "Page 2"
if st.sidebar.button("Page 3", key="page3"):
    st.session_state.page = "Page 3"


# ________________________________________________________________________________________________________________________
# Display content based on the selected page
if st.session_state.page == "Home":
    st.title("Home Page")
    st.write("Welcome to the Home page!")
# ________________________________________________________________________________________________________________________
elif st.session_state.page == "Credentials":
    st.title("Enter your credentials")

    def direct_to_page2():
        st.session_state.credentials_ok = True
        st.session_state.page = "Page 2"
        return

    # Text input fields
    username_input = st.text_input(
        "Enter your username:",
        value=st.session_state.username,  # Set value from session state
        placeholder="Type here..."
    )
    password_input = st.text_input(
        "Enter your password:",
        value=st.session_state.password,  # Set value from session state
        placeholder="Type here...",
        type='password'
    )

    # Update session state dynamically
    if username_input != st.session_state.username:
        st.session_state.username = username_input
    if password_input != st.session_state.password:
        st.session_state.password = password_input

    # Submit button
    if st.button("Submit"):
        check = check_amazon_credentials(st.session_state.username, st.session_state.password)
        if check == "Login Successful":
            st.success("Correct Username & Password")
            st.button("Go to Page 2", on_click=direct_to_page2)
        else:
            st.error(check)

    # Clear credentials button
    if st.button("Clear Credentials"):
        # Clear session state for username and password
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.credentials_ok = False
        st.success("Credentials have been cleared.")
# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Page 2") & (st.session_state.credentials_ok == True):
    st.title("Model Implementation")

    # Initialize session state if it's not already present with 4 empty strings
    if "urls" not in st.session_state:
        st.session_state.urls = []

    if "show_predict_button" not in st.session_state:
        st.session_state.show_predict_button = False

    def clear_url_data(index):
        st.session_state.urls[index] = ""

    def clean_urls():
        seen_urls = set()
        for i, url in enumerate(st.session_state.urls):
            if url in seen_urls and url != "":  # Check if the URL is a duplicate
                st.warning(f"Product URL {i + 1} is duplicate!")
                clear_url_data(i)
            elif url != "":  # Add unique, non-empty URLs to the set
                seen_urls.add(url)

        # Filter out empty strings from the URLs list
        st.session_state.urls = [url.strip() for url in st.session_state.urls if url.strip() != ""]
        st.session_state.global_state_urls = st.session_state.urls
        st.write(st.session_state.urls)
        st.session_state.show_predict_button = True


    # Create a single form to handle all input fields
    with st.form(key="url_form"):
        # Loop through and create 4 text input fields for URLs
        for i in range(4):
            st.session_state.urls.append("")
            # Set value for the URL text input
            st.session_state.urls[i] = st.text_input(f"Product URL {i + 1}:", value=st.session_state.urls[i],
                                                     key=f"product_url_{i}")

        # Submit button to submit the form
        submit_button = st.form_submit_button("Submit")

    # If the form is submitted, clean the URLs
    if submit_button:
        clean_urls()

    # Button to trigger the prediction (calls the clean_urls function)
    if st.session_state.show_predict_button:
        if st.button("Predict Ranking"):
            st.session_state.compute_signal = True
            st.session_state.page = "Page 3"

# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Page 2") & (st.session_state.credentials_ok == False):
    st.title("Model Implementation")
    st.subheader("Please enter and verify your credentials!!")
# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Page 3") & (st.session_state.credentials_ok == True):
    st.title("Results")

    st.session_state.global_state_urls = [url.strip() for url in st.session_state.global_state_urls if url.strip() is not ""]
    total_urls = len(st.session_state.global_state_urls)

    st.write(st.session_state.global_state_urls)

    # Check if product_details is already computed and stored
    if ('product_details' not in st.session_state) | st.session_state.compute_signal:
        product_details = []

        # Process each URL and store the results in product_details
        for i, url in enumerate(st.session_state.global_state_urls):
            # Simulate or replace this with your actual scraping function
            # product_details.append(scrape_reviews(url))  # Replace with actual function call
            product_details.append(scrape_reviews(url, i+1))  # This will fetch and process the reviews for each URL


        # Store the processed data in session state to avoid re-computation
        st.session_state.product_details = product_details
        st.session_state.compute_signal = False
    else:
        # Use previously computed results from session state
        product_details = st.session_state.product_details

    # Present the product details for each product in a nice format
    for details in product_details:
        # Create a layout with columns to organize the content
        col1, col2 = st.columns([2, 1])

        avg_star_rating = 0
        # Display the product name in the first column
        with col1:
            st.subheader(details["product_name"])  # Display product name as a header

        with col2:
            st.markdown("### 🌟 Star Ratings Breakdown:")
            if "5.0" in details['stars']:
                st.write("5 ⭐️ : ", details['stars'].count("5.0"))
                avg_star_rating += 5*details['stars'].count("5.0")
            if "4.0" in details['stars']:
                st.write("4 ⭐️ : ", details['stars'].count("4.0"))
                avg_star_rating += 4 * details['stars'].count("4.0")
            if "3.0" in details['stars']:
                st.write("3 ⭐️ : ", details['stars'].count("3.0"))
                avg_star_rating += 3 * details['stars'].count("3.0")
            if "2.0" in details['stars']:
                st.write("2 ⭐️ : ", details['stars'].count("2.0"))
                avg_star_rating += 2 * details['stars'].count("2.0")
            if "1.0" in details['stars']:
                st.write("1 ⭐️ : ", details['stars'].count("1.0"))
                avg_star_rating += 1 * details['stars'].count("1.0")

        avg_star_rating /= len(details['stars'])
        # Display the average rating if it's not a dictionary
        st.markdown(f"### 🌟 Average Rating: **{round(avg_star_rating,1)}** Stars")


        # Expandable section to show reviews
        with st.expander("Customer Reviews"):
            # Display reviews inside an expandable section for better UI
            for review in details["reviews"]:
                st.write(f"- {review}")


    #     # Append rankings
    #     st.session_state.rankings.append((url, avg_sentiment))
    #
    #     # Update progress bar
    #     progress.progress(int((i / total_urls) * 100))
    #
    #     # Display results for the current URL
    #     st.write(f"**URL {i}: {url}**")
    #     st.write(f"- Total Reviews: {num_reviews}")
    #     st.write("- Top 5 Reviews:")
    #     for review in top_reviews:
    #         st.write(f"  - {review}")
    #     st.write(f"- Predicted Average Sentiment: {avg_sentiment:.2f}")
    #
    # # Final results
    # if total_urls > 0:
    #     st.success("All URLs processed successfully!")
    #
    # # Display Rankings
    # st.subheader("Product Rankings by Average Sentiment")
    # sorted_rankings = sorted(st.session_state.rankings, key=lambda x: x[1], reverse=True)
    # for rank, (url, sentiment) in enumerate(sorted_rankings, start=1):
    #     st.write(f"{rank}. {url} - Average Sentiment: {sentiment:.2f}")

# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Page 3") & (st.session_state.credentials_ok == False):
    st.title("Results")
    st.subheader("No results found!!")
